import psutil
import os
import logging

log = logging.getLogger(__name__)

class ResourceMonitorAMD:
    def __init__(self):
        # Cambiato da card0 a card1
        self.base_path = "/sys/class/drm/card1/device"
        self.vram_total_path = os.path.join(self.base_path, "mem_info_vram_total")
        self.vram_used_path = os.path.join(self.base_path, "mem_info_vram_used")
        
        if not os.path.exists(self.vram_used_path):
            # Proviamo a vedere se è sotto renderD128 se card1 fallisce
            alternative_path = "/sys/class/drm/renderD128/device"
            if os.path.exists(os.path.join(alternative_path, "mem_info_vram_used")):
                self.base_path = alternative_path
                self.vram_total_path = os.path.join(self.base_path, "mem_info_vram_total")
                self.vram_used_path = os.path.join(self.base_path, "mem_info_vram_used")
                self.amd_enabled = True
            else:
                log.warning("Monitoraggio AMD non disponibile.")
                self.amd_enabled = False
        else:
            self.amd_enabled = True

    def get_stats(self):
        stats = {}
        
        # RAM di sistema (identica per tutti)
        ram = psutil.virtual_memory()
        stats['ram_used_gb'] = round(ram.used / (1024**3), 2)
        stats['ram_percent'] = ram.percent

        # VRAM AMD (Lettura diretta dal kernel)
        if self.amd_enabled:
            try:
                with open(self.vram_total_path, 'r') as f:
                    total = int(f.read().strip())
                with open(self.vram_used_path, 'r') as f:
                    used = int(f.read().strip())
                
                stats['vram_used_gb'] = round(used / (1024**3), 2)
                stats['vram_total_gb'] = round(total / (1024**3), 2)
                stats['vram_percent'] = round((used / total) * 100, 1)
            except Exception as e:
                log.error(f"Errore lettura VRAM AMD: {e}")
                stats['vram_used_gb'] = 0
        
        return stats

    def log_stats(self, context=""):
        s = self.get_stats()
        msg = f"📊 [STATS {context}] RAM: {s['ram_used_gb']}GB ({s['ram_percent']}%)"
        if self.amd_enabled:
            msg += f" | VRAM AMD: {s['vram_used_gb']}/{s['vram_total_gb']}GB ({s['vram_percent']}%)"
        
        log.info(msg)