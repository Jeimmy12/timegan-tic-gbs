# src/preprocessor.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from joblib import dump

class TimeGANPreprocessor:
    """
    Preprocesador que preserva características esenciales para diversidad:
    Características principales:
    - Preserva la integridad temporal de las trayectorias
    - Mantiene relaciones físicas entre posición y velocidad
    - Normaliza datos para optimizar el entrenamiento
    """

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.original_stats = {}

    # Carga y limpia datos de un ejercicio específico.
    def load_and_clean(self, filepath, etiqueta):
        """Carga y limpia datos de un ejercicio específico"""
        print(f"📊 Cargando {etiqueta}…")
        df = pd.read_csv(filepath)
        ejercicio = df[df['Etiqueta'] == etiqueta].copy()
        if ejercicio.empty:
            raise ValueError(f"No se encontraron datos para {etiqueta}")
        print(f"   • Datos originales: {len(ejercicio)} puntos")
        ejercicio_clean = ejercicio.dropna(
            subset=self.config.FEATURE_COLS + ['tt_01']
        )
        print(f"   • Después de limpieza: {len(ejercicio_clean)} puntos")
        return ejercicio_clean

    # Interpola trayectoria a longitud uniforme preservando suavidad.
    def interpolate_trajectory(self, df, target_len=None):
        """Interpolación suave que preserva características dinámicas"""
        if target_len is None:
            target_len = self.config.SEQ_LEN

        t_original = df['tt_01'].values
        t_min, t_max = t_original.min(), t_original.max()
        t_uniform = np.linspace(t_min, t_max, target_len)

        interpolated = {}

        for col in self.config.FEATURE_COLS:
            valores = df[col].values

            # Interpolación cúbica para suavidad
            if len(t_original) >= 4:
                f = interp1d(t_original, valores, kind='cubic',
                           fill_value='extrapolate', bounds_error=False)
            else:
                f = interp1d(t_original, valores, kind='linear',
                           fill_value='extrapolate', bounds_error=False)

            interpolated[col] = f(t_uniform)

            # Suavizado opcional para trayectorias muy ruidosas
            window_length = min(21, target_len // 10)
            if window_length % 2 == 0:
                window_length += 1

            if window_length >= 5:
                interpolated[col] = savgol_filter(
                    interpolated[col], window_length=window_length, polyorder=3
                )

        return interpolated

    def process_single_exercise(self, filepath, etiqueta):
        """Pipeline completo: carga → limpia → interpola → analiza → (aumenta si necesario)"""
        print(f"\n🏃 PROCESANDO {etiqueta}")
        
        # 1) Cargar y limpiar
        df_clean = self.load_and_clean(filepath, etiqueta)

        # 2) Interpolar y transformar a array
        interp_dict = self.interpolate_trajectory(df_clean)
        data = np.stack([interp_dict[c] for c in self.config.FEATURE_COLS], axis=1)
        #data = data[np.newaxis, :, :]  # [1, seq_len, features] si es 1 muestra

        # 3) Análisis de características
        self._analyze_exercise_characteristics(data, etiqueta)

        # 4) Aumentar si solo hay una muestra
        #if data.shape[0] == 1:
         #   print("🔁 Solo 1 muestra detectada — Generando aumentos para evitar sobreajuste...")
          #  data = self.augment_single_sequence(data[0], n_augmented=10)  # [10, seq_len, features]

        return data  # Siempre retorna [n_samples, seq_len, features]


    def _analyze_exercise_characteristics(self, data, etiqueta):
        pos_ranges = np.max(data[:, :3], axis=0) - np.min(data[:, :3], axis=0)
        vel_ranges = np.max(data[:, 3:], axis=0) - np.min(data[:, 3:], axis=0)

        print(f"   📐 Rangos de posición: {pos_ranges.round(3)}")
        print(f"   🏃 Rangos de velocidad: {vel_ranges.round(3)}")
        print(f"   📊 Variabilidad espacial: {np.mean(pos_ranges):.3f}")

        # Guardar características para referencia
        self.original_stats[etiqueta] = {
            'pos_ranges': pos_ranges,
            'vel_ranges': vel_ranges,
            'spatial_variability': np.mean(pos_ranges),
            'temporal_variability': np.mean(vel_ranges)
        }

    def augment_single_sequence(self, sequence, n_augmented=10, std_pos=0.05, std_vel=0.01):
        """
        Aumenta una secuencia aplicando ruido gaussiano diferenciado para posición y velocidad.
        Mantiene la coherencia física y evita distorsiones bruscas.
        """
        augmented = [sequence]
        for _ in range(n_augmented - 1):
            noise = np.zeros_like(sequence)
            # Ruido más suave para posición (columnas 0-2)
            noise[:, :3] = np.random.normal(0, std_pos, size=sequence[:, :3].shape)
            # Ruido aún más suave para velocidad (columnas 3-5)
            noise[:, 3:] = np.random.normal(0, std_vel, size=sequence[:, 3:].shape)

            noisy_seq = sequence + noise
            augmented.append(noisy_seq)

        return np.stack(augmented)


    


