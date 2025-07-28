# src/models.py

import torch
import torch.nn as nn
from torch.nn.functional import gelu, leaky_relu

class TemporalBlock(nn.Module):
    """Bloque temporal con atención para capturar dependencias complejas"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        # Mecanismo de auto-atención multi-cabeza
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        # Capa1 de normalización
        self.norm1 = nn.LayerNorm(hidden_dim)
        # Red feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        # Capa2 de normalización
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))

# === COMPONENTES DE TIMEGAN ===
class Embedder(nn.Module):
    """Encoder optimizado para capturar patrones de movimiento"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        # Proyección inicial
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        # LSTM para capturar dependencias temporales
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        # Bloques de atención temporal
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(hidden_dim, dropout) for _ in range(2)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        for block in self.temporal_blocks:
            x = block(x)
        return self.output_norm(x)


class Recovery(nn.Module):
    """Decoder que reconstruye trayectorias realistas manteniendo coherencia temporal y física."""
    def __init__(self, hidden_dim, output_dim, num_layers):
        super().__init__()
        # Bloques de atención temporal
        self.temporal_blocks = nn.ModuleList([TemporalBlock(hidden_dim) for _ in range(2)])
        # LSTM para secuencias de salida
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0)
        # Proyección final
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, h):
        for block in self.temporal_blocks:
            h = block(h)
        h, _ = self.lstm(h)
        return self.output_proj(h)


class Generator(nn.Module):
    """Generador con alta capacidad de variabilidad crea secuencias sintéticas desde ruido aleatorio."""
    def __init__(self, noise_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        # Proyección para capturar variabilidad
        self.input_proj = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        # LSTM generativo
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        # Bloques de atención
        self.temporal_blocks = nn.ModuleList([TemporalBlock(hidden_dim, dropout) for _ in range(2)])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, z):
        z = self.input_proj(z)
        z, _ = self.lstm(z)
        for block in self.temporal_blocks:
            z = block(z)
        return self.output_norm(z)


class Supervisor(nn.Module):
    """Supervisor para consistencia temporal, asegura que las transiciones temporales sean coherentes"""
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, max(1, num_layers-1),
                            batch_first=True,
                            dropout=0.2 if num_layers > 2 else 0)
        self.temporal_block = TemporalBlock(hidden_dim)

    def forward(self, h):
        s, _ = self.lstm(h)
        return self.temporal_block(s)


class Discriminator(nn.Module):
    """Discriminador robusto para distinguir secuencias reales de sintéticas."""
    def __init__(self, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        # LSTM discriminativo
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        # Atención para características relevantes
        self.temporal_blocks = nn.ModuleList([TemporalBlock(hidden_dim, dropout)])
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h):
        h, _ = self.lstm(h)
        for block in self.temporal_blocks:
            h = block(h)
        # Global max pooling para características robustas
        h = torch.max(h, dim=1)[0]  # pooling
        return self.classifier(h)


def create_models(config):
    """Instancia y despliega todos los submodelos de TimeGAN en GPU/CPU"""
    models = {
        'E': Embedder(len(config.FEATURE_COLS), config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT_RATE).to(config.DEVICE),
        'R': Recovery(config.HIDDEN_DIM, len(config.FEATURE_COLS), config.NUM_LAYERS).to(config.DEVICE),
        'G': Generator(config.NOISE_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT_RATE).to(config.DEVICE),
        'S': Supervisor(config.HIDDEN_DIM, config.NUM_LAYERS).to(config.DEVICE),
        'D': Discriminator(config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT_RATE).to(config.DEVICE),
    }
    total = sum(p.numel() for m in models.values() for p in m.parameters())
    print(f"✅ Modelos creados - Total parámetros: {total:,}")
    return models
