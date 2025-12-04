# Video Matte - Video Background Remover
# Copyright (c) 2025 Hadi
# Author: Hadi
# GitHub: https://github.com/lahadiyani
# Licensed under MIT License

import argparse
import cv2
import numpy as np
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import warnings

def check_model_exists(model_name="u2net"):
    """Cek apakah model sudah ada di lokal"""
    from rembg.session_factory import get_model_path
    
    try:
        model_path = get_model_path(model_name)
        return os.path.exists(model_path)
    except:
        # Fallback untuk lokasi default model
        home_dir = Path.home()
        model_path = home_dir / ".u2net" / f"{model_name}.onnx"
        return model_path.exists()

def download_model_with_progress(model_name="u2net"):
    """Download model dengan progress bar"""
    from rembg.session_factory import download_model
    
    print(f"\n📥 Downloading model '{model_name}'...")
    print("   (Hanya dilakukan sekali saat pertama kali)")
    
    # Create custom progress bar
    class DownloadProgressBar:
        def __init__(self, total_size):
            self.progress_bar = tqdm(
                total=total_size,
                desc="   Downloading",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                colour="yellow"
            )
            self.downloaded = 0
        
        def update(self, chunk_size):
            self.downloaded += chunk_size
            self.progress_bar.update(chunk_size)
        
        def close(self):
            self.progress_bar.close()
    
    # Monkey patch the download function to add progress bar
    import urllib.request
    original_urlretrieve = urllib.request.urlretrieve
    
    def urlretrieve_with_progress(url, filename, reporthook=None):
        def progress_hook(count, block_size, total_size):
            if progress_hook.pbar is None:
                progress_hook.pbar = DownloadProgressBar(total_size)
            downloaded = count * block_size
            if downloaded < total_size:
                progress_hook.pbar.update(block_size)
            else:
                progress_hook.pbar.close()
                progress_hook.pbar = None
        
        progress_hook.pbar = None
        return original_urlretrieve(url, filename, progress_hook)
    
    # Temporarily replace urlretrieve
    urllib.request.urlretrieve = urlretrieve_with_progress
    
    try:
        download_model(model_name)
        print(f"\n✅ Model '{model_name}' berhasil di-download!")
        return True
    except Exception as e:
        print(f"\n❌ Gagal mendownload model: {e}")
        return False
    finally:
        # Restore original function
        urllib.request.urlretrieve = original_urlretrieve

def check_cuda_availability():
    """Cek apakah CUDA tersedia di sistem"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def install_torch_with_cuda():
    """Instal PyTorch dengan CUDA support"""
    print("\n" + "="*60)
    print("MENGINSTAL PYTORCH DENGAN CUDA SUPPORT")
    print("="*60)
    
    # Cek sistem operasi dan Python version
    import platform
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    print(f"Sistem Operasi: {platform.system()}")
    print(f"Python Version: {python_version}")
    
    # Tentukan perintah instalasi berdasarkan platform
    if platform.system() == "Windows":
        command = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    elif platform.system() == "Linux":
        command = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    else:
        command = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ]
    
    print(f"\nMenjalankan perintah: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("\nInstalasi berhasil!")
        print("Output:", result.stdout[:500])
        
        # Verifikasi instalasi
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ GPU CUDA tersedia! Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("\n⚠️  GPU CUDA TIDAK terdeteksi setelah instalasi.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Gagal menginstal PyTorch dengan CUDA:")
        print(f"Error: {e.stderr}")
        return False

def setup_gpu_environment(use_gpu):
    """Setup environment untuk GPU jika diperlukan"""
    if use_gpu:
        print("\n🔍 Memeriksa ketersediaan GPU CUDA...")
        
        # Cek apakah torch sudah terinstal
        try:
            import torch
            torch_installed = True
        except ImportError:
            torch_installed = False
        
        if not torch_installed:
            print("PyTorch belum terinstal.")
            response = input("Apakah Anda ingin menginstal PyTorch dengan CUDA support? (y/n): ")
            if response.lower() == 'y':
                if install_torch_with_cuda():
                    print("\n✅ PyTorch dengan CUDA berhasil diinstal!")
                else:
                    print("\n⚠️  Menggunakan CPU karena instalasi gagal.")
                    return False
            else:
                print("Menggunakan CPU...")
                return False
        else:
            # Cek CUDA availability
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                print(f"✅ GPU CUDA tersedia!")
                print(f"   Device: {device_name}")
                print(f"   CUDA Version: {cuda_version}")
                
                # Set environment variable untuk memaksa penggunaan GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                return True
            else:
                print("❌ PyTorch terinstal tetapi CUDA tidak tersedia.")
                print("   Mungkin Anda perlu menginstal versi PyTorch dengan CUDA support.")
                response = input("Apakah Anda ingin mencoba menginstal ulang PyTorch dengan CUDA? (y/n): ")
                if response.lower() == 'y':
                    if install_torch_with_cuda():
                        return True
                    else:
                        print("⚠️  Menggunakan CPU karena CUDA tidak tersedia.")
                        return False
                else:
                    print("⚠️  Menggunakan CPU...")
                    return False
    else:
        # Nonaktifkan GPU untuk penggunaan CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return False

class VideoProcessor:
    """Class untuk memproses video dengan progress tracking"""
    
    def __init__(self, use_gpu=False, post_process=True):
        self.use_gpu = use_gpu
        self.post_process = post_process
        self.session = None
        self.start_time = None
        self.processed_frames = 0
        self.total_frames = 0
        self.fps_history = []
        
    def check_and_prepare_model(self):
        """Cek dan siapkan model sebelum memulai"""
        print("\n" + "="*60)
        print("🔄 PREPARASI MODEL")
        print("="*60)
        
        model_name = "u2net"
        model_exists = check_model_exists(model_name)
        
        if model_exists:
            print(f"✅ Model '{model_name}' ditemukan di cache lokal.")
            print("   Menggunakan model yang sudah ada...")
            return True
        else:
            print(f"⚠️  Model '{model_name}' tidak ditemukan di cache lokal.")
            response = input("   Apakah Anda ingin mendownload model? (y/n): ")
            
            if response.lower() == 'y':
                if download_model_with_progress(model_name):
                    return True
                else:
                    print("\n❌ Tidak dapat melanjutkan tanpa model.")
                    return False
            else:
                print("\n❌ Proses dibatalkan. Model diperlukan untuk segmentasi.")
                return False
    
    def initialize(self):
        """Inisialisasi model dan session"""
        print("\n" + "="*60)
        print("⚙️  INISIALISASI SISTEM")
        print("="*60)
        
        # Setup progress bar untuk inisialisasi
        with tqdm(total=4, desc="Inisialisasi", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            # Step 1: Import library
            pbar.set_description("📦 Import library")
            try:
                from rembg import remove, new_session
                self.rembg_remove = remove
                self.new_session = new_session
            except ImportError:
                print("\n❌ Library 'rembg' tidak ditemukan.")
                print("   Instal dengan: pip install rembg")
                sys.exit(1)
            pbar.update(1)
            time.sleep(0.1)
            
            # Step 2: Setup GPU
            pbar.set_description("⚙️  Setup GPU/CPU")
            self.gpu_available = setup_gpu_environment(self.use_gpu)
            pbar.update(1)
            time.sleep(0.1)
            
            # Step 3: Load model
            pbar.set_description("🤖 Load model segmentasi")
            
            # Suppress warnings dari rembg
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if self.gpu_available:
                    self.session = new_session("u2net")
                    print("\n🚀 Menggunakan GPU untuk pemrosesan...")
                else:
                    self.session = new_session("u2net")
                    print("\n🐌 Menggunakan CPU untuk pemrosesan...")
            
            pbar.update(1)
            time.sleep(0.1)
            
            # Step 4: Selesai
            pbar.set_description("✅ Inisialisasi selesai")
            pbar.update(1)
        
        print(f"\n{'='*60}")
        print(f"MODE: {'GPU 🚀' if self.gpu_available else 'CPU 💻'}")
        print(f"POST-PROCESSING: {'AKTIF ✅' if self.post_process else 'NONAKTIF ⚠️'}")
        print(f"{'='*60}")
    
    def process_frame(self, frame, width, height):
        """Proses satu frame video"""
        # Resize frame jika diperlukan
        if hasattr(self, 'resize_factor') and self.resize_factor != 1.0:
            frame = cv2.resize(frame, (width, height))
        
        # Konversi BGR ke RGB untuk rembg
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Remove background menggunakan rembg dengan session
        result = self.rembg_remove(pil_image, session=self.session)
        
        # Konversi ke numpy array
        result_np = np.array(result)
        
        # Pisahkan alpha channel (mask)
        if result_np.shape[2] == 4:  # RGBA
            mask = result_np[:, :, 3]
            frame_no_bg = result_np[:, :, :3]
        else:
            # Fallback jika tidak ada alpha channel
            mask = np.ones((height, width), dtype=np.uint8) * 255
            frame_no_bg = result_np
        
        # Konversi ke BGR
        frame_no_bg_bgr = cv2.cvtColor(frame_no_bg, cv2.COLOR_RGB2BGR)
        
        # Normalize mask ke 0-1
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_normalized = np.expand_dims(mask_normalized, axis=2)
        
        # Post-processing: smoothing mask
        if self.post_process:
            # Gunakan Gaussian blur untuk smooth mask
            mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
            mask_normalized = mask_blurred.astype(np.float32) / 255.0
            mask_normalized = np.expand_dims(mask_normalized, axis=2)
        
        return frame_no_bg_bgr, mask_normalized
    
    def process_video(self, input_path, output_path, background_color=(0, 0, 0), resize_factor=1.0):
        """Proses video lengkap"""
        self.start_time = time.time()
        self.resize_factor = resize_factor
        
        # Buka video input
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"\n❌ Error: Tidak dapat membuka video {input_path}")
            return False
        
        # Ambil properti video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Apply resize jika diperlukan
        if resize_factor != 1.0:
            width = int(width * resize_factor)
            height = int(height * resize_factor)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Tampilkan informasi video
        print(f"\n📊 INFORMASI VIDEO")
        print(f"   {'─' * 50}")
        print(f"   📁 Input: {os.path.basename(input_path)}")
        print(f"   💾 Output: {os.path.basename(output_path)}")
        print(f"   📐 Resolusi: {width}x{height}")
        print(f"   ⏱️  FPS: {fps}")
        print(f"   🎞️  Total frame: {self.total_frames:,}")
        print(f"   🎨 Background: RGB{background_color[::-1]}")
        print(f"   🔧 Resize factor: {resize_factor}")
        
        # Estimasi waktu
        if self.total_frames > 0:
            # Estimasi berdasarkan mode
            if self.gpu_available:
                fps_estimate = 8  # FPS estimasi untuk GPU
            else:
                fps_estimate = 2  # FPS estimasi untuk CPU
            
            estimated_time = (self.total_frames / fps_estimate) / 60
            print(f"   ⏳ Estimasi waktu: {estimated_time:.1f} menit")
        
        print(f"   {'─' * 50}")
        print("\n🎬 Memulai pemrosesan video...")
        
        # Progress bar utama dengan format kustom
        progress_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {percentage:3.0f}%]"
        
        # Progress bar utama
        with tqdm(total=self.total_frames, 
                 desc="🎬 Memproses Video",
                 bar_format=progress_bar_format,
                 unit="frame",
                 colour="green",
                 ncols=80,
                 mininterval=0.5) as pbar:
            
            frame_count = 0
            last_update_time = time.time()
            last_fps_update = time.time()
            fps_samples = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.processed_frames += 1
                
                # Proses frame
                try:
                    frame_no_bg_bgr, mask_normalized = self.process_frame(frame, width, height)
                except Exception as e:
                    print(f"\n⚠️  Error memproses frame {frame_count}: {e}")
                    continue
                
                # Buat background dengan warna yang diinginkan
                background = np.ones_like(frame_no_bg_bgr) * background_color
                
                # Gabungkan frame dengan background menggunakan mask
                mask_3ch = np.repeat(mask_normalized, 3, axis=2)
                frame_final = (frame_no_bg_bgr * mask_3ch + 
                              background * (1 - mask_3ch)).astype(np.uint8)
                
                # Tulis frame ke output
                out.write(frame_final)
                
                # Update progress bar setiap 0.5 detik atau setiap frame untuk video pendek
                current_time = time.time()
                time_since_last_update = current_time - last_update_time
                
                if time_since_last_update >= 0.5 or frame_count == self.total_frames:
                    # Hitung FPS
                    if time_since_last_update > 0:
                        current_fps = 1 / time_since_last_update
                        fps_samples.append(current_fps)
                        
                        # Keep only last 10 samples
                        if len(fps_samples) > 10:
                            fps_samples.pop(0)
                        
                        avg_fps = np.mean(fps_samples) if fps_samples else 0
                        
                        # Update deskripsi dengan informasi real-time
                        elapsed = current_time - self.start_time
                        remaining = (self.total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
                        
                        # Format status dengan informasi yang jelas
                        status_parts = []
                        status_parts.append(f"Frame {frame_count:,}/{self.total_frames:,}")
                        status_parts.append(f"FPS: {avg_fps:.1f}")
                        
                        # Tampilkan persentase
                        percentage = (frame_count / self.total_frames) * 100
                        status_parts.append(f"{percentage:.1f}%")
                        
                        # Tampilkan waktu tersisa
                        if remaining > 0:
                            if remaining < 60:
                                eta_str = f"{remaining:.0f}s"
                            elif remaining < 3600:
                                eta_str = f"{remaining/60:.1f}m"
                            else:
                                eta_str = f"{remaining/3600:.1f}h"
                            status_parts.append(f"ETA: {eta_str}")
                        
                        status = " | ".join(status_parts)
                        pbar.set_description(f"🎬 {status}")
                    
                    last_update_time = current_time
                
                pbar.update(1)
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return True
    
    def format_time(self, seconds):
        """Format waktu ke format yang mudah dibaca"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def print_summary(self, output_path):
        """Tampilkan summary setelah proses selesai"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        print(f"\n{'='*60}")
        print("✅ PROSES SELESAI")
        print(f"{'='*60}")
        
        print(f"\n📊 STATISTIK")
        print(f"   {'─' * 40}")
        print(f"   🎞️  Total frame: {self.processed_frames:,}")
        print(f"   ⏱️  Waktu total: {self.format_time(total_time)}")
        
        if total_time > 0:
            avg_fps = self.processed_frames / total_time
            print(f"   ⚡ Rata-rata FPS: {avg_fps:.2f}")
        
        print(f"   💾 File output: {output_path}")
        print(f"   📁 Ukuran file: {self.get_file_size(output_path)}")
        
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"\n🖥️  STATISTIK GPU")
                    print(f"   {'─' * 40}")
                    print(f"   💾 Memory used: {memory_allocated:.2f} GB")
                    print(f"   📊 Memory reserved: {memory_reserved:.2f} GB")
            except:
                pass
        
        print(f"\n✨ {self.get_random_success_message()}")
    
    def get_file_size(self, filepath):
        """Dapatkan ukuran file dalam format yang mudah dibaca"""
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        return "Unknown"
    
    def get_random_success_message(self):
        """Pesan sukses acak"""
        messages = [
            "Background berhasil dihapus! 🎉",
            "Video siap digunakan! ✨",
            "Proses selesai dengan sempurna! ✅",
            "Background removal sukses! 🚀",
            "Video sudah diproses! 👍",
            "Hasil luar biasa! Siap diedit lebih lanjut! 🎬"
        ]
        import random
        return random.choice(messages)

def parse_color_string(color_str):
    """Parse string warna ke tuple BGR"""
    try:
        # Format: "B,G,R" atau "R,G,B"
        colors = [int(c.strip()) for c in color_str.split(',')]
        if len(colors) == 3:
            # Asumsikan input adalah B,G,R
            return tuple(colors)
        else:
            raise ValueError
    except:
        print("Warning: Format warna tidak valid, menggunakan hitam (0,0,0)")
        return (0, 0, 0)

def main():
    parser = argparse.ArgumentParser(
        description='Hapus background dari video dan ganti dengan warna tertentu',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎬 CONTOH PENGGUNAAN:
  
  # Basic usage dengan CPU
  python video_bg_remove.py -i input.mp4 -o output.mp4
  
  # Dengan GPU (jika tersedia)
  python video_bg_remove.py -i input.mp4 -o output.mp4 --gpu
  
  # Dengan background merah dan GPU
  python video_bg_remove.py -i input.mp4 -o output.mp4 --background "255,0,0" --gpu
  
  # Dengan resize untuk percepatan dan GPU
  python video_bg_remove.py -i input.mp4 -o output.mp4 --resize 0.5 --gpu
  
  # Tanpa post-processing dengan GPU
  python video_bg_remove.py -i input.mp4 -o output.mp4 --no-post-process --gpu

📝 CATATAN:
  - Opsi --gpu akan mencoba menggunakan GPU jika CUDA tersedia
  - Jika PyTorch dengan CUDA belum terinstal, program akan menawarkan untuk menginstalnya
  - Warna background format: R,G,B (Red, Green, Blue)
  - Model hanya didownload sekali saat pertama kali digunakan
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Path ke video input')
    parser.add_argument('-o', '--output', required=True,
                       help='Path untuk video output')
    parser.add_argument('-b', '--background', default='0,0,0',
                       help='Warna background baru dalam format R,G,B (default: 0,0,0 hitam)')
    parser.add_argument('-r', '--resize', type=float, default=1.0,
                       help='Faktor resize untuk percepatan (0.1-1.0, default: 1.0)')
    parser.add_argument('--no-post-process', action='store_true',
                       help='Nonaktifkan post-processing smoothing')
    parser.add_argument('--gpu', action='store_true',
                       help='Gunakan GPU untuk pemrosesan (jika tersedia)')
    parser.add_argument('--skip-model-check', action='store_true',
                       help='Lewati pengecekan model (gunakan jika sudah yakin model ada)')
    
    args = parser.parse_args()
    
    # Validasi input
    if not os.path.exists(args.input):
        print(f"❌ Error: File input tidak ditemukan: {args.input}")
        return
    
    # Validasi resize factor
    if args.resize <= 0 or args.resize > 1:
        print("⚠️  Warning: Resize factor harus antara 0.1 dan 1.0. Menggunakan 1.0")
        args.resize = 1.0
    
    # Parse warna background (ubah dari R,G,B ke B,G,R untuk OpenCV)
    try:
        colors = [int(c.strip()) for c in args.background.split(',')]
        if len(colors) == 3:
            # Konversi dari R,G,B ke B,G,R
            bg_color = (colors[2], colors[1], colors[0])
        else:
            raise ValueError
    except:
        print("⚠️  Warning: Format warna tidak valid, menggunakan hitam (0,0,0)")
        bg_color = (0, 0, 0)
    
    # Tampilkan header
    print("\n" + "="*60)
    print("🎥 VIDEO BACKGROUND REMOVER")
    print("="*60)
    print(f"🕐 Waktu mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Buat instance processor
    processor = VideoProcessor(
        use_gpu=args.gpu,
        post_process=not args.no_post_process
    )
    
    # Cek model (kecuali dilewati)
    if not args.skip_model_check:
        if not processor.check_and_prepare_model():
            print("\n❌ Proses dibatalkan.")
            return
    
    # Inisialisasi
    processor.initialize()
    
    # Proses video
    print("\n" + "="*60)
    print("🎬 MEMPROSES VIDEO")
    print("="*60)
    
    success = processor.process_video(
        input_path=args.input,
        output_path=args.output,
        background_color=bg_color,
        resize_factor=args.resize
    )
    
    # Tampilkan summary
    if success:
        processor.print_summary(args.output)
    
    print(f"\n{'='*60}")
    print(f"🕐 Waktu selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()