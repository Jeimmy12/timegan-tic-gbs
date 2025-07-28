# src/trainer.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import time
from joblib import dump, load
import shutil
from models import create_models
from losses import TimeGANLosses
from preprocessor import TimeGANPreprocessor

class TimeGANTrainer:
    """Entrenador con enfoque en diversidad y estabilidad"""

    def __init__(self, config, models, losses):
        self.config = config
        self.models = models
        self.losses = losses
        self._setup_optimizers()
        
        self.train_history = {
            'd_loss': [], 'g_loss': [], 'correlation': [],
            'mse': [], 'diversity_score': []
        }
        self.best_score = -1
        self.patience_counter = 0

    def _setup_optimizers(self):
        """Configurar optimizadores con diferentes learning rates"""
        self.opt_er = optim.Adam(
            list(self.models['E'].parameters()) + list(self.models['R'].parameters()),
            lr=self.config.LEARNING_RATE_G, betas=(0.5, 0.999), weight_decay=1e-5
        )
        
        self.opt_gs = optim.Adam(
            list(self.models['G'].parameters()) + list(self.models['S'].parameters()),
            lr=self.config.LEARNING_RATE_G, betas=(0.5, 0.999), weight_decay=1e-5
        )

        self.opt_d = optim.Adam(
            self.models['D'].parameters(),
            lr=self.config.LEARNING_RATE_D, betas=(0.5, 0.999), weight_decay=1e-5
        )

        # Schedulers para convergencia estable
        self.scheduler_er = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_er, T_max=self.config.EPOCHS, eta_min=1e-6
        )
        self.scheduler_gs = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_gs, T_max=self.config.EPOCHS, eta_min=1e-6
        )
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_d, T_max=self.config.EPOCHS, eta_min=1e-7
        )

    def compute_gradient_penalty(self, real_data, fake_data):
        """Gradient penalty para estabilidad WGAN-GP"""
        batch_size = real_data.size(0)
        device = real_data.device

        alpha = torch.rand(batch_size, 1, 1, device=device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        d_interpolates = self.models['D'](interpolates)

        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.config.GRADIENT_PENALTY_WEIGHT * gradient_penalty

    def train_step(self, real_data, epoch):
        """Paso de entrenamiento optimizado"""
        batch_size = real_data.size(0)
        device = real_data.device

        # ========== ENTRENAR DISCRIMINADOR ==========
        for _ in range(self.config.D_STEPS):
            with torch.no_grad():
                H_real = self.models['E'](real_data)

            Z = torch.randn(batch_size, self.config.SEQ_LEN,
                           self.config.NOISE_DIM, device=device)
            with torch.no_grad():
                E_hat = self.models['G'](Z)
                H_hat = self.models['S'](E_hat)

            d_real = self.models['D'](H_real)
            d_fake = self.models['D'](H_hat)

            gp = self.compute_gradient_penalty(H_real, H_hat)
            d_loss = -torch.mean(d_real) + torch.mean(d_fake) + gp

            self.opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.models['D'].parameters(), self.config.GRADIENT_CLIP
            )
            self.opt_d.step()

        # ========== ENTRENAR GENERADOR ==========
        g_losses = []

        for _ in range(self.config.G_STEPS):
            Z = torch.randn(batch_size, self.config.SEQ_LEN,
                           self.config.NOISE_DIM, device=device)
            E_hat = self.models['G'](Z)
            H_hat = self.models['S'](E_hat)
            X_hat = self.models['R'](H_hat)

            with torch.no_grad():
                H_real = self.models['E'](real_data)

            # Pérdidas del generador
            d_fake = self.models['D'](H_hat)
            g_adv = -torch.mean(d_fake)
            g_sup = F.mse_loss(H_hat[:, :-1, :], E_hat[:, 1:, :])
            g_recon = self.losses.reconstruction_loss(real_data, X_hat)
            g_temp = self.losses.temporal_consistency_loss(X_hat)
            g_phys = self.losses.physics_consistency_loss(X_hat)
            g_div = self.losses.diversity_loss(X_hat)  # NUEVA!

            # Pérdida total balanceada
            g_loss = (g_adv + g_sup +
                     self.config.RECONSTRUCTION_WEIGHT * g_recon +
                     self.config.TEMPORAL_CONSISTENCY_WEIGHT * g_temp +
                     self.config.PHYSICS_CONSISTENCY_WEIGHT * g_phys +
                     self.config.DIVERSITY_WEIGHT * g_div)

            self.opt_gs.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.models['G'].parameters()) +
                list(self.models['S'].parameters()),
                self.config.GRADIENT_CLIP
            )
            self.opt_gs.step()

            g_losses.append(g_loss.item())

        # ========== ACTUALIZAR AUTOENCODER ==========
        H = self.models['E'](real_data)
        X_tilde = self.models['R'](H)

        ae_loss = (self.losses.reconstruction_loss(real_data, X_tilde) +
                  0.1 * self.losses.temporal_consistency_loss(X_tilde) +
                  0.1 * self.losses.physics_consistency_loss(X_tilde))

        self.opt_er.zero_grad()
        ae_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.models['E'].parameters()) +
            list(self.models['R'].parameters()),
            self.config.GRADIENT_CLIP
        )
        self.opt_er.step()

        return {
            'd_loss': d_loss.item(),
            'g_loss': np.mean(g_losses),
            'ae_loss': ae_loss.item()
        }

    def save_checkpoint(self, filename, is_best=False):
        """Guardar modelo"""
        checkpoint = {
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'optimizers': {
                'opt_er': self.opt_er.state_dict(),
                'opt_gs': self.opt_gs.state_dict(),
                'opt_d': self.opt_d.state_dict()
            },
            'train_history': self.train_history,
            'best_score': self.best_score,
            'config': self.config
        }

        filepath = os.path.join(self.config.OUTPUT_PATH, filename)
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(self.config.OUTPUT_PATH, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Nuevo mejor modelo guardado!")

    def load_checkpoint(self, filename='best_model.pt'):
        """Cargar modelo entrenado"""
        filepath = os.path.join(self.config.OUTPUT_PATH, filename)

        if not os.path.exists(filepath):
            print(f"❌ No se encontró {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.config.DEVICE, weights_only=False)

        for name, model in self.models.items():
            model.load_state_dict(checkpoint['models'][name])

        self.train_history = checkpoint['train_history']
        self.best_score = checkpoint['best_score']

        print(f"✅ Modelo cargado: {filename}")
        print(f"🏆 Mejor score: {self.best_score:.4f}")
        return True

def save_exercise(activo, mano, ejercicio, generator,
                    patients):
    """
    Guarda la imagen de evaluación y los CSVs generados
    para cada paciente sintético en la ruta relativa a pExercise_dir.
    """

    # 2. Mover imagen de evaluación
    img_name = f"evaluation_{activo}_{mano}_{ejercicio}.png"
    img_src  = generator.config.OUTPUT_PATH / img_name
    img_dst  = generator.config.IMG_DIR / img_name
    pExercise_dir = generator.config.PEXERCISE_DIR

    if img_src.exists():
        shutil.move(str(img_src), str(img_dst))
        print(f"🖼️ Imagen movida a: {img_dst}")
    else:
        print(f"⚠️ Imagen no encontrada: {img_src}")

    # 3. Guardar CSVs generados
    ruta_base = pExercise_dir / activo / mano / ejercicio
    ruta_base.mkdir(parents=True, exist_ok=True)

    for i, patient in enumerate(patients):
        df = pd.DataFrame(patient, columns=generator.config.FEATURE_COLS)
        df.insert(0, "time", range(len(patient)))
        nombre_csv = f"paciente{i+1}_{activo}_{mano}_{ejercicio}.csv"
        ruta_csv = ruta_base / nombre_csv
        df.to_csv(ruta_csv, index=False)
        print(f"✅ Guardado: {ruta_csv}")

def test_balanced_config(config, filepath, etiqueta):
    """
    Probar configuración balanceada: correlación alta + diversidad recuperada
    """
    from preprocessor import TimeGANPreprocessor
    from generator import DiverseDataGenerator 
    print("🧪 PRUEBA CON CONFIGURACIÓN BALANCEADA")
    print("="*60)
    print("🎯 Objetivo: Correlación 0.60+ Y Diversidad 0.60+")
    print("="*60)

    # Aplicar configuración balanceada
    preprocessor = TimeGANPreprocessor(config)

    try:
        # Procesar datos
        data_processed = preprocessor.process_single_exercise(filepath, etiqueta)
        data_normalized = preprocessor.scaler.fit_transform(data_processed)
        data_tensor = torch.tensor(data_normalized[np.newaxis, ...], dtype=torch.float32)

        # Crear dataset y trainer
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        trainer = create_trainer(config, data_processed)

        # Entrenamiento balanceado (menos epochs que el optimizado)
        print(f"\n🏋️ ENTRENAMIENTO BALANCEADO...")

        # Pre-entrenamiento moderado
        print("📚 Pre-entrenamiento balanceado...")
        for epoch in range(40):  # Menos que la versión optimizada
            for (real_data,) in dataloader:
                real_data = real_data.to(config.DEVICE)

                H = trainer.models['E'](real_data)
                X_tilde = trainer.models['R'](H)

                # Pérdidas balanceadas
                recon_loss = trainer.losses.reconstruction_loss(real_data, X_tilde)
                temp_loss = trainer.losses.temporal_consistency_loss(X_tilde)
                phys_loss = trainer.losses.physics_consistency_loss(X_tilde)
                div_loss = trainer.losses.diversity_loss(X_tilde)  # ¡Incluir diversidad!

                loss = (config.RECONSTRUCTION_WEIGHT * recon_loss +
                       config.TEMPORAL_CONSISTENCY_WEIGHT * temp_loss +
                       config.PHYSICS_CONSISTENCY_WEIGHT * phys_loss +
                       config.DIVERSITY_WEIGHT * div_loss)

                trainer.opt_er.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(trainer.models['E'].parameters()) +
                    list(trainer.models['R'].parameters()),
                    config.GRADIENT_CLIP
                )
                trainer.opt_er.step()

            if epoch % 15 == 0:
                print(f"   Epoch {epoch}: Loss {loss.item():.4f}")

        # Pre-entrenamiento supervisor balanceado
        print("📚 Pre-entrenamiento supervisor balanceado...")
        for epoch in range(40):
            for (real_data,) in dataloader:
                real_data = real_data.to(config.DEVICE)

                with torch.no_grad():
                    H = trainer.models['E'](real_data)

                H_sup = trainer.models['S'](H)
                loss = F.mse_loss(H[:, 1:, :], H_sup[:, :-1, :])

                trainer.opt_gs.zero_grad()
                loss.backward()
                trainer.opt_gs.step()
            if epoch % 15 == 0:
                print(f"   Epoch {epoch}: Loss {loss.item():.4f}")

        # Entrenamiento adversarial balanceado
        print("⚔️ Entrenamiento adversarial balanceado...")
        for epoch in range(120):  # Menos epochs para evitar overfitting
            for (real_data,) in dataloader:
                real_data = real_data.to(config.DEVICE)
                metrics = trainer.train_step(real_data, epoch)

            if epoch % 30 == 0:
                print(f"   Epoch {epoch}: D={metrics['d_loss']:.4f}, G={metrics['g_loss']:.4f}")

        # Crear generador diverso mejorado
        enhanced_generator = DiverseDataGenerator(trainer, preprocessor, config)
        enhanced_generator.trainer = trainer
        enhanced_generator.preprocessor = preprocessor

        # Generar muestras con máxima diversidad
        samples_generated = []
        with torch.no_grad():
            for i in range(7):  # Más muestras para probar diversidad
                strategy_idx = i % len(enhanced_generator.generation_strategies)
                strategy = enhanced_generator.generation_strategies[strategy_idx]

                sample = enhanced_generator._generate_candidate_with_strategy(i, i*10, strategy)
                samples_generated.append(sample)

        # Análisis con visualización mejorada
        real_denorm = preprocessor.scaler.inverse_transform(data_normalized)
        metrics = DiverseDataGenerator(trainer, preprocessor, config).create_enhanced_quick_visualization(
            real_denorm, samples_generated[:5], f"{etiqueta}_balanced_test"
        )

        print(f"CONFIGURACIÓN    | CORRELACIÓN | DIVERSIDAD | BALANCE")
        print("-" * 60)
        print(f"Balanceada       | {metrics['avg_correlation']:.3f}       | {metrics['avg_diversity']:.3f}      | {(metrics['avg_correlation'] + metrics['avg_diversity'])/2:.3f}")
        print("-" * 60)

        # Evaluar éxito del balance
        balance_score = (metrics['avg_correlation'] + metrics['avg_diversity']) / 2

        print(f"\n🎯 EVALUACIÓN DEL BALANCE:")
        if metrics['avg_correlation'] > 0.70 and metrics['avg_diversity'] > 0.55:
            print(f"   🎉 ¡ÉXITO COMPLETO! Balance óptimo alcanzado")
            print(f"   ✅ Correlación: {metrics['avg_correlation']:.3f} > 0.70")
            print(f"   ✅ Diversidad: {metrics['avg_diversity']:.3f} > 0.55")
            print(f"   ✅ Score balance: {balance_score:.3f}")
            print(f"   🚀 PERFECTO PARA TESIS")
        elif metrics['avg_correlation'] > 0.65:
            print(f"   ✅ BUEN balance alcanzado")
            print(f"   ✅ Correlación: {metrics['avg_correlation']:.3f}")
            print(f"   ⚠️ Diversidad: {metrics['avg_diversity']:.3f}")
            print(f"   📊 Score balance: {balance_score:.3f}")
            print(f"   👍 ACEPTABLE PARA TESIS")
        else:
            print(f"   ⚠️ Balance necesita ajuste")
            print(f"   📊 Continuar experimentando con parámetros")

        return {
            'correlation': metrics['avg_correlation'],
            'diversity': metrics['avg_diversity'],
            'balance_score': balance_score,
            'trainer': trainer,
            'enhanced_generator': enhanced_generator
        }

    except Exception as e:
        print(f"❌ Error en prueba balanceada: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_model(config, filepath, activo, mano, ejercicio):
    from preprocessor import TimeGANPreprocessor
    from trainer import create_trainer    
    """
    Entrena un modelo TimeGAN completo y guarda modelo, imagen y CSVs.

    Parámetros:
    - config: instancia de TimeGANConfig
    - filepath: ruta al CSV real (por ejemplo: realData/activos1_derecha.csv)
    - activo: string como 'activos1'
    - mano: 'derecha' o 'izquierda'
    - ejercicio: 'Ejercicio1', 'Ejercicio2', etc.
    - n_pacientes: cuántos pacientes sintéticos generar (se planificaron 9)
    """
    etiqueta = f"{activo}_{mano}_{ejercicio}"
    print(f"\n🚀 ENTRENANDO MODELO FINAL PARA {activo}_{mano}_{ejercicio}")
    print("=" * 60)

    # === 1. Preprocesamiento ===
    preprocessor = TimeGANPreprocessor(config)
    data = preprocessor.process_single_exercise(filepath, ejercicio)
    data_normalized = preprocessor.scaler.fit_transform(data)
    data_tensor = torch.tensor(data_normalized[np.newaxis, ...], dtype=torch.float32)

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # === 2. Crear entrenador ===
    trainer = create_trainer(config, data)

    # === 3. Para guardar las img de pérdidas ===
    ae_losses, sup_losses, adv_d_losses, adv_g_losses = [], [], [], []
    
    # === 4. Entrenamiento completo ===
    # Pre-entrenamiento autoencoder
    print(f"\n📚 Pre-entrenamiento Autoencoder ({config.PRETRAIN_EPOCHS} epochs)")
    for epoch in range(config.PRETRAIN_EPOCHS):
        for (real_data,) in dataloader:
            real_data = real_data.to(config.DEVICE)

            H = trainer.models['E'](real_data)
            X_tilde = trainer.models['R'](H)

            recon_loss = trainer.losses.reconstruction_loss(real_data, X_tilde)
            temp_loss = trainer.losses.temporal_consistency_loss(X_tilde)
            phys_loss = trainer.losses.physics_consistency_loss(X_tilde)
            div_loss = trainer.losses.diversity_loss(X_tilde)
            
            loss = (config.RECONSTRUCTION_WEIGHT * recon_loss +
                    config.TEMPORAL_CONSISTENCY_WEIGHT * temp_loss +
                    config.PHYSICS_CONSISTENCY_WEIGHT * phys_loss +
                    config.DIVERSITY_WEIGHT * div_loss)

            trainer.opt_er.zero_grad(); loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(trainer.models['E'].parameters()) +
                list(trainer.models['R'].parameters()),
                config.GRADIENT_CLIP
            )
            trainer.opt_er.step()
            ae_losses.append(loss.item())

        if epoch % 30 == 0:
            print(f"   Epoch {epoch}: AE Loss = {loss.item():.4f}")

    # Pre-entrenamiento supervisor
    print(f"\n📚 Pre-entrenamiento Supervisor ({config.PRETRAIN_EPOCHS} epochs)")
    for epoch in range(config.PRETRAIN_EPOCHS):
        for (real_data,) in dataloader:
            real_data = real_data.to(config.DEVICE)
            with torch.no_grad():
                H = trainer.models['E'](real_data)
            H_sup = trainer.models['S'](H)
            loss = torch.nn.functional.mse_loss(H[:, 1:, :], H_sup[:, :-1, :])

            trainer.opt_gs.zero_grad(); loss.backward(); trainer.opt_gs.step()
            sup_losses.append(loss.item())

        if epoch % 30 == 0:
            print(f"   Epoch {epoch}: S Loss = {loss.item():.4f}")        

    # Entrenamiento adversarial
    print(f"\n⚔️ Entrenamiento Adversarial Final ({config.EPOCHS} epochs)")
    for epoch in range(config.EPOCHS):
        for (real_data,) in dataloader:
            real_data = real_data.to(config.DEVICE)
            metrics = trainer.train_step(real_data, epoch)
            
            adv_d_losses.append(metrics['d_loss'])
            adv_g_losses.append(metrics['g_loss'])

        if epoch % 50 == 0:
            print(f"   Epoch {epoch}: D={metrics['d_loss']:.4f}, G={metrics['g_loss']:.4f}")

    
    ckpt_name = f"timegan_{etiqueta}.pt"
    trainer.save_checkpoint(ckpt_name, is_best=True)

    # === 6. Visualización de pérdidas ===
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(ae_losses)
    axs[0].set_title("Autoencoder Pretraining Loss")
    axs[1].plot(sup_losses)
    axs[1].set_title("Supervisor Pretraining Loss")
    axs[2].plot(adv_d_losses, label="Discriminator")
    axs[2].plot(adv_g_losses, label="Generator")
    axs[2].set_title("Adversarial Training Losses")
    axs[2].legend()
    plt.suptitle(f"Visualización de pérdidas {etiqueta}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    loss_plot_path = os.path.join(config.IMG_DIR, f"losses_{etiqueta}.png")
    plt.savefig(loss_plot_path)
    plt.show()
    plt.close()
    print(f"📉 Imagen de pérdidas guardada en: {loss_plot_path}")

    return trainer, preprocessor, data, data_normalized


# Crear entrenador
def create_trainer(config, exercise_data=None):
    """Crear entrenador completo"""
    models = create_models(config)
    losses = TimeGANLosses(config)
    trainer = TimeGANTrainer(config, models, losses)

    if exercise_data is not None:
        print(f"✅ Entrenador listo para datos de shape: {exercise_data.shape}")

    return trainer