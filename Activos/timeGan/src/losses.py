import torch
import torch.nn.functional as F

class TimeGANLosses:
    """
    Funciones de pérdida optimizadas para datos sintéticos.
    
    Incluye pérdidas específicas para:
    - Reconstrucción fiel de trayectorias
    - Consistencia temporal en movimientos
    - Coherencia física posición-velocidad
    - Diversidad entre muestras generadas
    """

    def __init__(self, config):
        self.config = config

    def reconstruction_loss(self, real_seq, fake_seq):
        """Pérdida de reconstrucción balanceada (L1 + ½·L2) combina L1 (robustez a outliers) y L2 (suavidad) para
        reconstrucción óptima de trayectorias."""
        l1 = F.l1_loss(fake_seq, real_seq)
        l2 = F.mse_loss(fake_seq, real_seq)
        return l1 + 0.5 * l2

    def temporal_consistency_loss(self, seq):
        """Consistencia temporal - suavidad de movimiento"""
        # Velocidades instantáneas
        vel = seq[:, 1:, :] - seq[:, :-1, :]
        # Aceleraciones
        acc = vel[:, 1:, :] - vel[:, :-1, :]

        # Penalizar cambios bruscos
        vel_smoothness = torch.mean(torch.abs(vel))
        acc_smoothness = torch.mean(torch.abs(acc))

        return vel_smoothness + 0.5 * acc_smoothness

    def physics_consistency_loss(self, seq):
        """Consistencia física: relación posición-velocidad"""
        if seq.size(2) < 6:  # Necesitamos pos + vel
            return torch.tensor(0.0, device=seq.device)

        # Posiciones y velocidades
        pos = seq[:, :, :3]  # x, y, z
        vel_given = seq[:, :, 3:]  # velocidades dadas

        # Velocidad calculada desde posiciones
        vel_computed = pos[:, 1:, :] - pos[:, :-1, :]
        vel_actual = vel_given[:, :-1, :]

        return F.mse_loss(vel_computed, vel_actual)

    def diversity_loss(self, generated_batch):
        """Pérdida que fomenta diversidad dentro del batch"""
        batch_size = generated_batch.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=generated_batch.device)

        # Calcular similitudes entre muestras del batch
        similarities = []
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                sample_i = generated_batch[i].flatten()
                sample_j = generated_batch[j].flatten()

                # Similitud coseno
                cos_sim = F.cosine_similarity(sample_i.unsqueeze(0),
                                            sample_j.unsqueeze(0))
                similarities.append(torch.abs(cos_sim))

        if similarities:
            # Penalizar alta similitud
            avg_similarity = torch.mean(torch.stack(similarities))
            return avg_similarity  # Queremos minimizar esto

        return torch.tensor(0.0, device=generated_batch.device)
