from pathlib import Path
import torch

class TimeGANConfig:
    """
    Configuración optimizada para generar datos sintéticos diversos.
    Esta clase contiene todos los hiperparámetros críticos para el modelo TimeGAN,
    optimizados específicamente para datos de trayectorias del movimiento.
    """
    
    # === PARÁMETROS DE DATOS ===
    FEATURE_COLS = ['pxr_01', 'pyr_01', 'pzr_01', 'vel1_01', 'vel2_01', 'vel3_01'] #Filtro de columnas
    SEQ_LEN = 200  # Longitud de secuencia temporal optimizada
    
    # === ARQUITECTURA DE RED ===
    # Controlan el tamaño y profundidad de LSTM/GRU en embedder (E), recovery (R), generator (G), supervisor (S) y discriminator (D).
    HIDDEN_DIM = 160        # Dimensión oculta balanceada
    NUM_LAYERS = 3          # Evita overfitting con datasets pequeños
    # Dimensionalidad del vector z de entrada al generador.
    NOISE_DIM = 96          # Dimensión de ruido para máxima variabilidad

    DROPOUT_RATE = 0.18      # Regularización moderada
    
    # === PARÁMETROS DE ENTRENAMIENTO ===
    # Tasas de aprendizaje independientes
    LEARNING_RATE_G = 1.5e-4   # Learning rate del generador
    LEARNING_RATE_D = 3e-5   # Learning rate del discriminador (más conservador)
    # Número total de pasadas y tamaño de lote
    EPOCHS = 250            # Épocas de entrenamiento
    BATCH_SIZE = 1           # Tamaño de batch adaptado a datos disponibles
    
    # === PESOS DE FUNCIONES DE PÉRDIDA ===
    GRADIENT_PENALTY_WEIGHT = 10.0     # Estabilidad WGAN-GP
    RECONSTRUCTION_WEIGHT = 2.5        # Fidelidad de reconstrucción
    TEMPORAL_CONSISTENCY_WEIGHT = 1.8  # Suavidad temporal
    PHYSICS_CONSISTENCY_WEIGHT = 1.2   # Coherencia física
    DIVERSITY_WEIGHT = 0.5            # Promoción de diversidad
    
    # === CONTROL DE ENTRENAMIENTO ===
    # Número de actualizaciones por ciclo
    G_STEPS = 4              # Pasos del generador por iteración
    D_STEPS = 1              # Pasos del discriminador por iteración
    # Fases iniciales de entrenamiento del Autoencoder (E/R)
    PRETRAIN_EPOCHS = 100    # Épocas de pre-entrenamiento
    GRADIENT_CLIP = 0.6
    EARLY_STOPPING_PATIENCE = 200    # Detengo si no mejora
    
    # === PARÁMETROS DE DIVERSIDAD ===
    MIN_DISTANCE_THRESHOLD = 0.25    # Umbral mínimo de distancia entre muestras
    SIMILARITY_THRESHOLD = 0.75      # Umbral máximo de similitud (similaridad coseno máxima)
    MAX_GENERATION_ATTEMPTS = 600    # Intentos máximos de generación
    GENERATION_DIVERSITY_WEIGHT = 0.7

    # === CONFIGURACIÓN TÉCNICA ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def __init__(self):
        # Base directory de todo el paquete timeGan
        self.BASE_DIR = Path(__file__).resolve().parents[2]

        # Directorios específicos
        self.CKPT_DIR   = self.BASE_DIR / "timeGan" / "checkpoints"
        self.IMG_DIR    = self.BASE_DIR / "timeGan" / "images"
        self.OUTPUT_PATH = self.CKPT_DIR

        self.PEXERCISE_DIR  = self.BASE_DIR / "syntheticData" / "pExercise"
        self.PPATIENT_DIR   = self.BASE_DIR / "syntheticData" / "pPatient"

        # Crear carpetas si no existen
        for d in (self.CKPT_DIR, self.IMG_DIR, self.PEXERCISE_DIR, self.PPATIENT_DIR):
            d.mkdir(parents=True, exist_ok=True)