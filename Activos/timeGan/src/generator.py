import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

from models import create_models
from losses import TimeGANLosses
from preprocessor import TimeGANPreprocessor
from trainer      import TimeGANTrainer

class DiverseDataGenerator:
    """
    🎭 GENERADOR DE DATOS DIVERSOS
    Funciona independientemente del entrenamiento
    """

    def __init__(self, trainer, preprocessor, config):
        self.trainer = trainer
        self.preprocessor = preprocessor
        self.config = config
        self.diversity_weight = self.config.GENERATION_DIVERSITY_WEIGHT


        # Estrategias de generación más variadas
        self.generation_strategies = [
            'high_noise_variation',      # NUEVO
            'frequency_modulation',      # NUEVO
            'temporal_phase_shift',      # NUEVO
            'amplitude_scaling',         # NUEVO
            'noise_interpolation',
            'hybrid_approach',
            'combo_modulation_shift'  
        ]

    def load_trained_model(self, model_path="best_model.pt"):
        """Cargar modelo ya entrenado"""

        if self.trainer is None:
            # Crear trainer y modelos
            models = create_models(self.config)
            losses = TimeGANLosses(self.config)
            self.trainer = TimeGANTrainer(self.config, models, losses)

        success = self.trainer.load_checkpoint(model_path)
        if success:
            print(f"✅ Modelo cargado para generación")
            print(f"🏆 Score del modelo: {self.trainer.best_score:.4f}")
            return True
        else:
            print(f"❌ No se pudo cargar modelo: {model_path}")
            return False

    def generate_diverse_patients(self, n_patients=10,
                                identifier="diverse_set",
                                quality_threshold=0.25):
        """
        🎭 GENERAR PACIENTES DIVERSOS
        Función principal de generación independiente
        """

        if self.trainer is None:
            print("❌ Necesitas cargar un modelo entrenado primero")
            print("   Usa: generator.load_trained_model('timegan_trained.pt')")
            return None, None

        print(f"\n🎭 GENERANDO {n_patients} PACIENTES DIVERSOS")
        print("="*60)
        print(f"🎯 Configuración de diversidad:")
        print(f"   • Distancia mínima: {self.config.MIN_DISTANCE_THRESHOLD}")
        print(f"   • Similitud máxima: {self.config.SIMILARITY_THRESHOLD}")
        print(f"   • Intentos máximos: {self.config.MAX_GENERATION_ATTEMPTS}")
        print(f"   • Umbral de calidad: {quality_threshold}")
        print("="*60)

        # Poner modelos en evaluación
        for model in self.trainer.models.values():
            model.eval()

        diverse_patients = []
        generation_metrics = []
        failed_generations = 0

        with torch.no_grad():
            for patient_id in range(n_patients):
                print(f"\n👤 Generando Paciente {patient_id + 1}/{n_patients}...")

                best_patient = None
                best_metrics = None
                best_score = -1

                for attempt in range(self.config.MAX_GENERATION_ATTEMPTS):
                    # Usar diferentes estrategias según el intento
                    strategy_idx = attempt % len(self.generation_strategies)
                    strategy = self.generation_strategies[strategy_idx]

                    # Generar candidato
                    candidate = self._generate_candidate_with_strategy(
                        patient_id, attempt, strategy
                    )

                    # Evaluar calidad y diversidad
                    metrics = self._evaluate_candidate(candidate, diverse_patients)

                    # Score combinado
                    combined_score = ((1 - self.diversity_weight) * metrics['realism_score'] +
                                    self.diversity_weight * metrics['diversity_score'])

                    # Verificar criterios de aceptación
                    is_quality_ok = metrics['realism_score'] >= quality_threshold
                    is_diverse_enough = metrics['is_diverse']

                    if combined_score > best_score:
                        best_patient = candidate
                        best_score = combined_score
                        best_metrics = metrics.copy()
                        best_metrics.update({
                            'combined_score': combined_score,
                            'attempts_needed': attempt + 1,
                            'strategy_used': strategy
                        })

                    # Early stopping si encontramos algo muy bueno
                    if combined_score > 0.85 and is_quality_ok and is_diverse_enough:
                        print(f"   ⚡ Converged early at attempt {attempt + 1}")
                        break

                # Evaluar resultado
                if best_patient is not None:
                    if best_metrics['realism_score'] >= 0.6:  # umbral mínimo aceptable
                        diverse_patients.append(best_patient)
                        generation_metrics.append(best_metrics)

                        print(f"   ✅ ÉXITO - Paciente {patient_id + 1} generado")
                        print(f"      📊 Realismo: {best_metrics['realism_score']:.3f}")
                        print(f"      🎭 Diversidad: {best_metrics['diversity_score']:.3f}")
                        print(f"      🎯 Score final: {best_metrics['combined_score']:.3f}")
                        print(f"      🔄 Intentos: {best_metrics['attempts_needed']}")
                        print(f"      🛠️ Estrategia: {best_metrics['strategy_used']}")
                    else:
                        print("   ⚠️ Paciente descartado por realismo muy bajo")
                else:
                    failed_generations += 1
                    print(f"   ❌ FALLO - No se pudo generar paciente {patient_id + 1}")
                    print(f"      💡 Intenta reducir quality_threshold o aumentar max_attempts")

        # Resumen final
        success_rate = len(diverse_patients) / n_patients

        print(f"\n🎉 GENERACIÓN COMPLETADA")
        print("="*60)
        print(f"✅ Pacientes generados exitosamente: {len(diverse_patients)}/{n_patients}")
        print(f"📊 Tasa de éxito: {success_rate:.1%}")
        print(f"❌ Generaciones fallidas: {failed_generations}")

        if len(diverse_patients) > 0:
            avg_realism = np.mean([m['realism_score'] for m in generation_metrics])
            avg_diversity = np.mean([m['diversity_score'] for m in generation_metrics])
            avg_attempts = np.mean([m['attempts_needed'] for m in generation_metrics])

            print(f"\n📈 MÉTRICAS PROMEDIO:")
            print(f"   Realismo: {avg_realism:.3f}")
            print(f"   Diversidad: {avg_diversity:.3f}")
            print(f"   Intentos promedio: {avg_attempts:.1f}")

            # Análisis de diversidad global
            diversity_analysis = self._analyze_global_diversity(diverse_patients)
            print(f"\n🎭 ANÁLISIS DE DIVERSIDAD GLOBAL:")
            print(f"   Similitud promedio entre pacientes: {diversity_analysis['avg_similarity']:.3f}")
            print(f"   Distancia promedio: {diversity_analysis['avg_distance']:.3f}")

            if diversity_analysis['avg_similarity'] < 0.3:
                print("   🎉 EXCELENTE diversidad - Pacientes muy diferentes")
            elif diversity_analysis['avg_similarity'] < 0.5:
                print("   ✅ BUENA diversidad - Adecuado para aplicaciones")
            else:
                print("   ⚠️ MODERADA diversidad - Considera ajustar parámetros")

        return diverse_patients, generation_metrics


    def _evaluate_candidate(self, candidate, existing_patients):
        """Evaluar calidad y diversidad de candidato"""

        # 1. EVALUAR REALISMO
        realism_score = self._evaluate_realism(candidate)

        # 2. EVALUAR DIVERSIDAD
        diversity_score, is_diverse = self._evaluate_diversity(candidate, existing_patients)

        # 3. MÉTRICAS ADICIONALES
        smoothness_score = self._evaluate_smoothness(candidate)
        physics_score = self._evaluate_physics_consistency(candidate)

        return {
            'realism_score': realism_score,
            'diversity_score': diversity_score,
            'is_diverse': is_diverse,
            'smoothness_score': smoothness_score,
            'physics_score': physics_score
        }

    def _evaluate_realism(self, sample):
        """Evaluar realismo de la muestra"""

        # 1. Rangos de posición razonables
        pos_ranges = np.max(sample[:, :3], axis=0) - np.min(sample[:, :3], axis=0)
        pos_score = np.mean([min(r/15.0, 1.0) for r in pos_ranges])  # Normalizado a [0,1]

        # 2. Rangos de velocidad razonables
        vel_ranges = np.max(sample[:, 3:], axis=0) - np.min(sample[:, 3:], axis=0)
        vel_score = np.mean([min(r/10.0, 1.0) for r in vel_ranges])

        # 3. Suavidad de trayectoria
        smoothness = self._evaluate_smoothness(sample)

        # 4. Consistencia física
        physics = self._evaluate_physics_consistency(sample)

        # Score total balanceado
        realism = (0.3 * pos_score + 0.2 * vel_score +
                  0.3 * smoothness + 0.2 * physics)

        return realism

    def _evaluate_smoothness(self, sample):
        """Evaluar suavidad de la trayectoria"""

        # Velocidades instantáneas
        velocities = np.diff(sample[:, :3], axis=0)
        # Aceleraciones
        accelerations = np.diff(velocities, axis=0)

        # Penalizar cambios bruscos
        vel_variability = np.mean(np.std(velocities, axis=0))
        acc_variability = np.mean(np.std(accelerations, axis=0))

        # Convertir a score (menos variabilidad = más suave = mejor)
        smoothness = 1.0 / (1.0 + vel_variability + 0.5 * acc_variability)

        return smoothness

    def _evaluate_physics_consistency(self, sample):
        """Evaluar consistencia física posición-velocidad"""

        if sample.shape[1] < 6:
            return 0.5  # Score neutro si no hay suficientes variables

        # Velocidad calculada desde posiciones
        computed_vel = np.diff(sample[:, :3], axis=0)
        given_vel = sample[:-1, 3:]

        # Correlación entre velocidades
        correlations = []
        for i in range(3):
            if (np.std(computed_vel[:, i]) > 1e-8 and
                np.std(given_vel[:, i]) > 1e-8):
                corr = np.corrcoef(computed_vel[:, i], given_vel[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        physics_score = np.mean(correlations) if correlations else 0.3
        return physics_score

    def _evaluate_diversity(self, candidate, existing_patients):
        """Evaluar diversidad respecto a pacientes existentes"""

        if len(existing_patients) == 0:
            return 1.0, True

        similarities = []
        distances = []

        for existing in existing_patients:
            # Similitud de coseno por variable
            var_similarities = []
            for i in range(candidate.shape[1]):
                cand_flat = candidate[:, i].reshape(1, -1)
                exist_flat = existing[:, i].reshape(1, -1)

                if (np.std(cand_flat) > 1e-8 and np.std(exist_flat) > 1e-8):
                    sim = cosine_similarity(cand_flat, exist_flat)[0, 0]
                    var_similarities.append(abs(sim))

            avg_similarity = np.mean(var_similarities) if var_similarities else 0
            similarities.append(avg_similarity)

            # Distancia euclidiana normalizada
            distance = np.mean(np.sqrt(np.sum((candidate - existing)**2, axis=1)))
            distances.append(distance)

        # Evaluar criterios de diversidad
        max_similarity = max(similarities)
        min_distance = min(distances)

        is_diverse = (max_similarity < self.config.SIMILARITY_THRESHOLD and
                     min_distance > self.config.MIN_DISTANCE_THRESHOLD)

        # Score de diversidad (distancia mínima normalizada)
        diversity_score = min(min_distance / (self.config.MIN_DISTANCE_THRESHOLD * 2), 1.0)

        return diversity_score, is_diverse

    def _analyze_global_diversity(self, patients):
        """Analizar diversidad global del conjunto"""

        n_patients = len(patients)
        if n_patients < 2:
            return {'avg_similarity': 0, 'avg_distance': 0}

        similarities = []
        distances = []

        for i in range(n_patients):
            for j in range(i+1, n_patients):
                # Similitud promedio por variable
                var_similarities = []
                for k in range(patients[i].shape[1]):
                    if (np.std(patients[i][:, k]) > 1e-8 and
                        np.std(patients[j][:, k]) > 1e-8):
                        corr = np.corrcoef(patients[i][:, k], patients[j][:, k])[0, 1]
                        if not np.isnan(corr):
                            var_similarities.append(abs(corr))

                if var_similarities:
                    similarities.append(np.mean(var_similarities))

                # Distancia euclidiana
                distance = np.mean(np.sqrt(np.sum((patients[i] - patients[j])**2, axis=1)))
                distances.append(distance)

        return {
            'avg_similarity': np.mean(similarities) if similarities else 0,
            'avg_distance': np.mean(distances) if distances else 0,
            'min_similarity': np.min(similarities) if similarities else 0,
            'max_distance': np.max(distances) if distances else 0
        }

    def save_diverse_patients(self, patients, metrics, identifier="diverse_set"):
        """Guardar pacientes diversos en archivos CSV"""

        print(f"\n💾 GUARDANDO {len(patients)} PACIENTES DIVERSOS...")

        saved_files = []

        for i, (patient, metric) in enumerate(zip(patients, metrics)):
            # Crear DataFrame
            df = pd.DataFrame(patient, columns=self.config.FEATURE_COLS)
            df['time'] = np.linspace(0, len(patient)-1, len(patient))

            # Nombre descriptivo del archivo
            filename = (f"patient_{identifier}_P{i+1:02d}_"
                       f"score_{metric['combined_score']:.3f}.csv")

            filepath = os.path.join(self.config.OUTPUT_PATH, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            saved_files.append(filename)

            print(f"   👤 Paciente {i+1}: {filename}")
            print(f"      📊 Realismo: {metric['realism_score']:.3f} | "
                  f"🎭 Diversidad: {metric['diversity_score']:.3f} | "
                  f"🎯 Score: {metric['combined_score']:.3f}")

        print(f"\n✅ {len(saved_files)} archivos guardados en: {self.config.OUTPUT_PATH}")
        return saved_files
    
    def _generate_candidate_with_strategy(self, patient_id, attempt, strategy):
        """Generar candidato usando estrategia específica (modo agresivo)."""
        # 1) Ruido base
        Z = torch.randn(1, self.config.SEQ_LEN, self.config.NOISE_DIM).to(self.config.DEVICE)

        if strategy == 'high_noise_variation':
            # Variación extrema de escala de ruido
            scale = 0.5 + 1.5 * np.sin(patient_id * 3.14 + attempt * 0.05)
            Z = Z * scale

        elif strategy == 'frequency_modulation':
            # Modulación de frecuencia con fases variables
            t = torch.linspace(0, 8*np.pi, self.config.SEQ_LEN).to(self.config.DEVICE)
            freq1 = 1 + patient_id * 0.3
            freq2 = 2 + patient_id * 0.7
            phase1 = attempt * 0.1
            phase2 = attempt * 0.15
            mod1 = 1 + 0.4 * torch.sin(freq1 * t + phase1)
            mod2 = 1 + 0.3 * torch.cos(freq2 * t + phase2)
            Z = Z * (mod1 * mod2).unsqueeze(0).unsqueeze(-1)

        elif strategy == 'temporal_phase_shift':
            # Desplaza el ruido circularmente en el tiempo
            shift = int((patient_id * 17 + attempt * 3) % self.config.SEQ_LEN)
            Z_shifted = torch.roll(Z, shifts=shift, dims=1)
            alpha = 0.3 + 0.4 * (attempt / self.config.MAX_GENERATION_ATTEMPTS)
            Z = (1 - alpha) * Z + alpha * Z_shifted

        elif strategy == 'amplitude_scaling':
            # Escala cada dimensión de ruido de forma distinta
            for dim in range(self.config.NOISE_DIM):
                factor = 0.6 + 0.8 * np.sin(patient_id * 0.5 + dim * 0.3 + attempt * 0.02)
                Z[:, :, dim] *= factor

        elif strategy == 'noise_interpolation':
            # Interpolación entre dos ruidos base
            Z2 = torch.randn_like(Z)
            alpha = 0.3 + 0.4 * (attempt / self.config.MAX_GENERATION_ATTEMPTS)
            Z = (1 - alpha) * Z + alpha * Z2

        elif strategy == 'hybrid_approach':
            # Mezcla de varias técnicas
            # parte de high_noise
            scale = 0.9 + 0.2 * np.sin(patient_id + attempt * 0.1)
            Z = Z * scale
            # + componente temporal ligero
            t = torch.linspace(0, 2*np.pi, self.config.SEQ_LEN).to(self.config.DEVICE)
            Z = Z + 0.2 * torch.sin(2*t + patient_id).unsqueeze(0).unsqueeze(-1)

        elif strategy == 'combo_modulation_shift':
            # Paso 1: frecuencia modulada
            t = torch.linspace(0, 8*np.pi, self.config.SEQ_LEN).to(self.config.DEVICE)
            freq1 = 1 + patient_id * 0.3
            freq2 = 2 + patient_id * 0.7
            phase1 = attempt * 0.1
            phase2 = attempt * 0.15
            mod1 = 1 + 0.4 * torch.sin(freq1 * t + phase1)
            mod2 = 1 + 0.3 * torch.cos(freq2 * t + phase2)
            Z = Z * (mod1 * mod2).unsqueeze(0).unsqueeze(-1)

            # Paso 2: desplazamiento temporal
            shift = int((patient_id * 17 + attempt * 3) % self.config.SEQ_LEN)
            alpha = 0.3 + 0.4 * (attempt / self.config.MAX_GENERATION_ATTEMPTS)
            Z_shifted = torch.roll(Z, shifts=shift, dims=1)
            Z = (1 - alpha) * Z + alpha * Z_shifted

        else:
            # Fallback a 'high_noise_variation' si se pide algo inesperado
            scale = 0.5 + 1.5 * np.sin(patient_id * 3.14 + attempt * 0.05)
            Z = Z * scale

        # 2) Genera muestra con tu red entrenada
        E_hat = self.trainer.models['G'](Z)
        H_hat = self.trainer.models['S'](E_hat)
        X_hat = self.trainer.models['R'](H_hat)

        sample = X_hat.cpu().numpy()[0]

        # 3) Desnormaliza si tienes scaler
        if self.preprocessor is not None:
            sample = self.preprocessor.scaler.inverse_transform(sample)

        return sample


    def create_enhanced_quick_visualization(self, real_data, generated_samples, etiqueta):
        """
        Visualización mejorada con más proyecciones y análisis de diversidad
        """

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        time_axis = np.arange(generated_samples[0].shape[0])
        colors = [
            'red', 'green', 'blue', 'orange', 'purple',
            'brown', 'pink', 'gray', 'olive', 'cyan',
            'magenta', 'gold', 'lime', 'navy', 'teal'
        ]


        # ========== FILA 1: PROYECCIONES ESPACIALES ==========

        # 1. Proyección XY
        ax = axes[0, 0]
        ax.plot(real_data[:, 0], real_data[:, 1], 'b-', label='Real', linewidth=3, alpha=0.8)
        for i, sample in enumerate(generated_samples):
            ax.plot(sample[:, 0], sample[:, 1], '--', color=colors[i],
                alpha=0.8, linewidth=2, label=f'Gen {i+1}')

        # Marcar inicio y fin
        ax.scatter(real_data[0, 0], real_data[0, 1], c='blue', s=100, marker='o',
                edgecolor='white', linewidth=2, label='Inicio Real')
        ax.scatter(real_data[-1, 0], real_data[-1, 1], c='blue', s=100, marker='s',
                edgecolor='white', linewidth=2, label='Fin Real')

        for i, sample in enumerate(generated_samples):
            ax.scatter(sample[0, 0], sample[0, 1], c=colors[i], s=80, marker='o', alpha=0.7)
            ax.scatter(sample[-1, 0], sample[-1, 1], c=colors[i], s=80, marker='s', alpha=0.7)

        ax.set_title('Proyección XY - Trayectorias', fontweight='bold', fontsize=12)
        ax.set_xlabel('Posición X')
        ax.set_ylabel('Posición Y')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 2. Proyección XZ (NUEVA!)
        ax = axes[0, 1]
        ax.plot(real_data[:, 0], real_data[:, 2], 'b-', label='Real', linewidth=3, alpha=0.8)
        for i, sample in enumerate(generated_samples):
            ax.plot(sample[:, 0], sample[:, 2], '--', color=colors[i],
                alpha=0.8, linewidth=2, label=f'Gen {i+1}')

        # Marcar inicio y fin
        ax.scatter(real_data[0, 0], real_data[0, 2], c='blue', s=100, marker='o',
                edgecolor='white', linewidth=2)
        ax.scatter(real_data[-1, 0], real_data[-1, 2], c='blue', s=100, marker='s',
                edgecolor='white', linewidth=2)

        for i, sample in enumerate(generated_samples):
            ax.scatter(sample[0, 0], sample[0, 2], c=colors[i], s=80, marker='o', alpha=0.7)
            ax.scatter(sample[-1, 0], sample[-1, 2], c=colors[i], s=80, marker='s', alpha=0.7)

        ax.set_title('Proyección XZ - Trayectorias', fontweight='bold', fontsize=12)
        ax.set_xlabel('Posición X')
        ax.set_ylabel('Posición Z')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 3. Proyección YZ
        ax = axes[0, 2]
        ax.plot(real_data[:, 1], real_data[:, 2], 'b-', label='Real', linewidth=3, alpha=0.8)
        for i, sample in enumerate(generated_samples):
            ax.plot(sample[:, 1], sample[:, 2], '--', color=colors[i],
                alpha=0.8, linewidth=2, label=f'Gen {i+1}')

        ax.scatter(real_data[0, 1], real_data[0, 2], c='blue', s=100, marker='o',
                edgecolor='white', linewidth=2)
        ax.scatter(real_data[-1, 1], real_data[-1, 2], c='blue', s=100, marker='s',
                edgecolor='white', linewidth=2)

        for i, sample in enumerate(generated_samples):
            ax.scatter(sample[0, 1], sample[0, 2], c=colors[i], s=80, marker='o', alpha=0.7)
            ax.scatter(sample[-1, 1], sample[-1, 2], c=colors[i], s=80, marker='s', alpha=0.7)

        ax.set_title('Proyección YZ - Trayectorias', fontweight='bold', fontsize=12)
        ax.set_xlabel('Posición Y')
        ax.set_ylabel('Posición Z')
        ax.grid(True, alpha=0.3)

        # 4. Trayectoria 3D
        ax = axes[0, 3]
        ax.remove()  # Eliminar el subplot 2D
        ax = fig.add_subplot(3, 4, 4, projection='3d')

        ax.plot(real_data[:, 0], real_data[:, 1], real_data[:, 2], 'b-',
            label='Real', linewidth=3, alpha=0.8)

        for i, sample in enumerate(generated_samples):
            ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], '--',
                color=colors[i], alpha=0.8, linewidth=2, label=f'Gen {i+1}')

        ax.scatter(real_data[0, 0], real_data[0, 1], real_data[0, 2],
                c='blue', s=100, marker='o')
        ax.scatter(real_data[-1, 0], real_data[-1, 1], real_data[-1, 2],
                c='blue', s=100, marker='s')

        ax.set_title('Trayectoria 3D', fontweight='bold', fontsize=12)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        # ========== FILA 2: SERIES TEMPORALES ==========

        # 5. Posición X vs Tiempo
        ax = axes[1, 0]
        ax.plot(time_axis, real_data[:, 0], 'b-', label='Real X', linewidth=3, alpha=0.8)
        for i, sample in enumerate(generated_samples):
            ax.plot(time_axis, sample[:, 0], '--', color=colors[i],
                alpha=0.8, linewidth=2, label=f'Gen {i+1}')
        ax.set_title('Posición X vs Tiempo', fontweight='bold', fontsize=12)
        ax.set_xlabel('Tiempo (muestras)')
        ax.set_ylabel('Posición X')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Posición Y vs Tiempo
        ax = axes[1, 1]
        ax.plot(time_axis, real_data[:, 1], 'b-', label='Real Y', linewidth=3, alpha=0.8)
        for i, sample in enumerate(generated_samples):
            ax.plot(time_axis, sample[:, 1], '--', color=colors[i],
                alpha=0.8, linewidth=2, label=f'Gen {i+1}')
        ax.set_title('Posición Y vs Tiempo', fontweight='bold', fontsize=12)
        ax.set_xlabel('Tiempo (muestras)')
        ax.set_ylabel('Posición Y')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 7. Posición Z vs Tiempo (NUEVA!)
        ax = axes[1, 2]
        ax.plot(time_axis, real_data[:, 2], 'b-', label='Real Z', linewidth=3, alpha=0.8)
        for i, sample in enumerate(generated_samples):
            ax.plot(time_axis, sample[:, 2], '--', color=colors[i],
                alpha=0.8, linewidth=2, label=f'Gen {i+1}')
        ax.set_title('Posición Z vs Tiempo', fontweight='bold', fontsize=12)
        ax.set_xlabel('Tiempo (muestras)')
        ax.set_ylabel('Posición Z')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 8. Magnitud de Velocidad
        ax = axes[1, 3]
        real_vel_mag = np.linalg.norm(real_data[:, 3:], axis=1)
        ax.plot(time_axis, real_vel_mag, 'b-', label='Real |Vel|', linewidth=3, alpha=0.8)

        for i, sample in enumerate(generated_samples):
            vel_mag = np.linalg.norm(sample[:, 3:], axis=1)
            ax.plot(time_axis, vel_mag, '--', color=colors[i],
                alpha=0.8, linewidth=2, label=f'Gen {i+1}')

        ax.set_title('Magnitud de Velocidad vs Tiempo', fontweight='bold', fontsize=12)
        ax.set_xlabel('Tiempo (muestras)')
        ax.set_ylabel('|Velocidad|')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ========== FILA 3: ANÁLISIS ESTADÍSTICO Y DIVERSIDAD ==========

        # 9. Correlaciones por muestra (mejorado)
        ax = axes[2, 0]
        correlations = []
        mse_scores = []

        for i, sample in enumerate(generated_samples):
            # Correlación promedio
            corrs = []
            for j in range(len(self.config.FEATURE_COLS)):
                if np.std(real_data[:, j]) > 1e-8 and np.std(sample[:, j]) > 1e-8:
                    corr = np.corrcoef(real_data[:, j], sample[:, j])[0, 1]
                    if not np.isnan(corr):
                        corrs.append(abs(corr))

            avg_corr = np.mean(corrs) if corrs else 0
            mse = np.mean((real_data - sample)**2)

            correlations.append(avg_corr)
            mse_scores.append(mse)

        x_pos = range(len(correlations))
        bars = ax.bar(x_pos, correlations, alpha=0.7, color=colors[:len(correlations)])
        ax.set_title('Correlación vs Datos Reales', fontweight='bold', fontsize=12)
        ax.set_xlabel('Muestra Generada')
        ax.set_ylabel('Correlación Promedio')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Gen {i+1}' for i in range(len(correlations))])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Agregar valores encima de las barras
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')

        # 10. Análisis de Diversidad Entre Muestras (NUEVO!)
        ax = axes[2, 1]

        if len(generated_samples) > 1:
            diversity_matrix = np.zeros((len(generated_samples), len(generated_samples)))

            for i in range(len(generated_samples)):
                for j in range(len(generated_samples)):
                    if i != j:
                        # Distancia euclidiana normalizada
                        dist = np.mean(np.sqrt(np.sum((generated_samples[i] - generated_samples[j])**2, axis=1)))
                        diversity_matrix[i, j] = dist
                    else:
                        diversity_matrix[i, j] = 0

            im = ax.imshow(diversity_matrix, cmap='viridis', interpolation='nearest')
            ax.set_title('Diversidad Entre Muestras\n(Más claro = Más diversas)', fontweight='bold', fontsize=12)
            ax.set_xlabel('Muestra')
            ax.set_ylabel('Muestra')
            ax.set_xticks(range(len(generated_samples)))
            ax.set_yticks(range(len(generated_samples)))
            ax.set_xticklabels([f'Gen {i+1}' for i in range(len(generated_samples))])
            ax.set_yticklabels([f'Gen {i+1}' for i in range(len(generated_samples))])

            # Agregar valores numéricos
            for i in range(len(generated_samples)):
                for j in range(len(generated_samples)):
                    text = ax.text(j, i, f'{diversity_matrix[i, j]:.2f}',
                                ha="center", va="center", color="white", fontsize=10)

            plt.colorbar(im, ax=ax, label='Distancia')

            # Calcular diversidad promedio
            upper_triangle = diversity_matrix[np.triu_indices(len(generated_samples), k=1)]
            avg_diversity = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0

            ax.text(0.02, 0.98, f'Diversidad\nPromedio:\n{avg_diversity:.3f}',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                facecolor="yellow", alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'Se necesitan\nal menos 2 muestras\npara análisis\nde diversidad',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)

        # 11. Distribuciones comparativas (mejorado)
        ax = axes[2, 2]

        # Crear histogramas para posición X como ejemplo
        ax.hist(real_data[:, 0], bins=25, alpha=0.6, color='blue',
            label='Real X', density=True, histtype='step', linewidth=3)

        for i, sample in enumerate(generated_samples):
            ax.hist(sample[:, 0], bins=25, alpha=0.4, color=colors[i],
                label=f'Gen {i+1} X', density=True, histtype='stepfilled')

        ax.set_title('Distribución Posición X\n(Comparación Real vs Sintético)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Valor Posición X')
        ax.set_ylabel('Densidad')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 12. Métricas de Calidad Resumidas (NUEVO!)
        ax = axes[2, 3]
        ax.axis('off')  # Sin ejes para mostrar solo texto

        # Calcular métricas resumidas
        avg_correlation = np.mean(correlations)
        avg_mse = np.mean(mse_scores)

        # Evaluar diversidad
        if len(generated_samples) > 1:
            diversity_scores = []
            for i in range(len(generated_samples)):
                for j in range(i+1, len(generated_samples)):
                    dist = np.mean(np.abs(generated_samples[i] - generated_samples[j]))
                    diversity_scores.append(dist)
            avg_diversity_score = np.mean(diversity_scores)
        else:
            avg_diversity_score = 0

        # Rangos de los datos
        real_range = np.max(real_data, axis=0) - np.min(real_data, axis=0)
        avg_real_range = np.mean(real_range[:3])  # Solo posiciones

        synth_ranges = []
        for sample in generated_samples:
            sample_range = np.max(sample, axis=0) - np.min(sample, axis=0)
            synth_ranges.append(np.mean(sample_range[:3]))
        avg_synth_range = np.mean(synth_ranges)

        range_similarity = 1 - abs(avg_real_range - avg_synth_range) / avg_real_range

        # Texto de resumen
        summary_text = f"""
    RESUMEN DE MÉTRICAS

    🎯 CALIDAD DEL MODELO:
    • Correlación Promedio: {avg_correlation:.3f}
    • MSE Promedio: {avg_mse:.4f}
    • Similitud de Rangos: {range_similarity:.3f}

    🎭 DIVERSIDAD:
    • Score de Diversidad: {avg_diversity_score:.3f}
    • Número de Muestras: {len(generated_samples)}

    📊 EVALUACIÓN:"""

        if avg_correlation > 0.6:
            summary_text += "\n✅ BUENA Calidad"
        elif avg_correlation > 0.4:
            summary_text += "\n⚠️ MODERADA Calidad"
        else:
            summary_text += "\n❌ BAJA Calidad"

        if avg_diversity_score > 1.0:
            summary_text += "\n✅ BUENA Diversidad"
        elif avg_diversity_score > 0.5:
            summary_text += "\n⚠️ MODERADA Diversidad"
        else:
            summary_text += "\n❌ BAJA Diversidad"

        if avg_correlation > 0.6:
            summary_text += "\n\n💡 RECOMENDACIÓN:\n   Proceder con\n   entrenamiento completo"
        else:
            summary_text += "\n\n💡 RECOMENDACIÓN:\n   Ajustar configuración\n   antes de continuar"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        # Título principal mejorado
        plt.suptitle(f'Evaluación Completa - {etiqueta}\n'
                    f'Correlación: {avg_correlation:.3f} | Diversidad: {avg_diversity_score:.3f} | '
                    f'Muestras: {len(generated_samples)}',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Guardar
        output_file = os.path.join(self.config.OUTPUT_PATH, f'enhanced_evaluation_{etiqueta}.png')
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"📊 Evaluación mejorada guardada: {output_file}")

        plt.show()

        return {
            'avg_correlation': avg_correlation,
            'avg_mse': avg_mse,
            'avg_diversity': avg_diversity_score,
            'range_similarity': range_similarity
        }

def generate_outputs(config, trainer, preprocessor, data_normalized, activo, mano, ejercicio, n_pacientes):
    from generator import DiverseDataGenerator
    from trainer import TimeGANTrainer, save_exercise
    from losses import TimeGANLosses

    etiqueta = f"{activo}_{mano}_{ejercicio}"
    generator = DiverseDataGenerator(trainer, preprocessor, config)
    real_denorm = preprocessor.scaler.inverse_transform(data_normalized)
    patients, _ = generator.generate_diverse_patients(n_patients=n_pacientes)
    metrics = generator.create_enhanced_quick_visualization(real_denorm, patients, etiqueta=etiqueta + "_final")

    # Guardar todo
    save_exercise(activo, mano, ejercicio, generator, patients)
