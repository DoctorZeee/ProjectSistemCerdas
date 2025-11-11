import sqlite3
from datetime import datetime
import json

class TBCChatbot:
    def __init__(self):
        self.setup_database()
        self.current_user = {}
        self.symptoms_score = 0
        
    def setup_database(self):
        """Setup database SQLite"""
        self.conn = sqlite3.connect('tbc_detection.db')
        self.cursor = self.conn.cursor()
        
        # Tabel users
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nama TEXT NOT NULL,
                usia INTEGER,
                jenis_kelamin TEXT,
                lokasi TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabel consultations
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                gejala TEXT,
                faktor_risiko TEXT,
                skor_total INTEGER,
                jenis_tbc TEXT,
                status_deteksi TEXT,
                rekomendasi TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        self.conn.commit()
        
    def start_chat(self):
        """Mulai percakapan chatbot"""
        print("\n" + "="*60)
        print("🏥 CHATBOT DETEKSI GEJALA TBC")
        print("="*60)
        print("\n⚠️  DISCLAIMER:")
        print("Sistem ini BUKAN pengganti diagnosis medis profesional.")
        print("Hasil deteksi hanya sebagai panduan awal.")
        print("Segera konsultasi ke dokter/puskesmas untuk pemeriksaan lanjutan.")
        print("="*60 + "\n")
        
        input("Tekan Enter untuk melanjutkan...")
        
        # Kumpulkan data dasar
        self.collect_basic_info()
        
        # Tanyakan gejala
        symptoms = self.ask_symptoms()
        
        # Tanyakan faktor risiko
        risk_factors = self.ask_risk_factors()
        
        # Analisis hasil
        result = self.analyze_results(symptoms, risk_factors)
        
        # Simpan ke database
        self.save_consultation(symptoms, risk_factors, result)
        
        # Tampilkan statistik
        self.show_statistics()
        
    def collect_basic_info(self):
        """Kumpulkan informasi dasar pengguna"""
        print("\n📋 DATA DIRI")
        print("-" * 60)
        
        nama = input("Nama lengkap: ")
        usia = int(input("Usia: "))
        print("\nJenis kelamin:")
        print("1. Laki-laki")
        print("2. Perempuan")
        jk = input("Pilih (1/2): ")
        jenis_kelamin = "Laki-laki" if jk == "1" else "Perempuan"
        lokasi = input("Kota/Kabupaten: ")
        
        # Simpan ke database
        self.cursor.execute('''
            INSERT INTO users (nama, usia, jenis_kelamin, lokasi)
            VALUES (?, ?, ?, ?)
        ''', (nama, usia, jenis_kelamin, lokasi))
        self.conn.commit()
        
        self.current_user = {
            'id': self.cursor.lastrowid,
            'nama': nama,
            'usia': usia,
            'jenis_kelamin': jenis_kelamin,
            'lokasi': lokasi
        }
        
    def ask_symptoms(self):
        """Tanyakan gejala-gejala TBC"""
        print("\n\n🔍 DETEKSI GEJALA")
        print("-" * 60)
        print("Jawab dengan Ya (y) atau Tidak (t)\n")
        
        symptoms = {}
        
        # Gejala utama TBC Paru
        questions = [
            ("batuk_lama", "Apakah Anda mengalami batuk lebih dari 2-3 minggu?", 3),
            ("batuk_darah", "Apakah batuk disertai darah?", 4),
            ("demam", "Apakah mengalami demam (terutama sore/malam)?", 2),
            ("keringat_malam", "Apakah berkeringat banyak di malam hari?", 2),
            ("penurunan_bb", "Apakah mengalami penurunan berat badan tanpa sebab jelas?", 3),
            ("nafsu_makan", "Apakah nafsu makan menurun?", 1),
            ("sesak_napas", "Apakah mengalami sesak napas?", 2),
            ("nyeri_dada", "Apakah merasakan nyeri dada?", 2),
            ("lemah_lesu", "Apakah merasa lemah dan lesu berkepanjangan?", 1)
        ]
        
        for key, question, score in questions:
            answer = input(f"{question} (y/t): ").lower()
            symptoms[key] = answer == 'y'
            if symptoms[key]:
                self.symptoms_score += score
                
        # Gejala TBC Ekstra Paru
        print("\n--- Gejala Tambahan (TBC Ekstra Paru) ---")
        
        extra_questions = [
            ("benjolan_leher", "Apakah ada benjolan di leher/ketiak?", 2),
            ("nyeri_tulang", "Apakah ada nyeri tulang/sendi yang persisten?", 2),
            ("bengkak_sendi", "Apakah ada pembengkakan pada sendi?", 2),
            ("sakit_kepala", "Apakah mengalami sakit kepala hebat/terus-menerus?", 2)
        ]
        
        for key, question, score in extra_questions:
            answer = input(f"{question} (y/t): ").lower()
            symptoms[key] = answer == 'y'
            if symptoms[key]:
                self.symptoms_score += score
                
        return symptoms
        
    def ask_risk_factors(self):
        """Tanyakan faktor risiko TBC"""
        print("\n\n⚠️  FAKTOR RISIKO")
        print("-" * 60)
        
        risk_factors = {}
        
        questions = [
            ("kontak_tbc", "Apakah pernah kontak dekat dengan penderita TBC?", 3),
            ("riwayat_tbc", "Apakah pernah menderita TBC sebelumnya?", 3),
            ("hiv", "Apakah memiliki HIV/AIDS atau gangguan sistem imun?", 4),
            ("diabetes", "Apakah memiliki diabetes?", 2),
            ("merokok", "Apakah Anda perokok aktif?", 1),
            ("lingkungan_padat", "Apakah tinggal di lingkungan padat/kurang ventilasi?", 1),
            ("gizi_buruk", "Apakah mengalami kekurangan gizi?", 2)
        ]
        
        for key, question, score in questions:
            answer = input(f"{question} (y/t): ").lower()
            risk_factors[key] = answer == 'y'
            if risk_factors[key]:
                self.symptoms_score += score
                
        return risk_factors
        
    def analyze_results(self, symptoms, risk_factors):
        """Analisis hasil dan tentukan jenis TBC"""
        print("\n\n📊 HASIL ANALISIS")
        print("=" * 60)
        
        # Tentukan jenis TBC
        jenis_tbc = []
        
        # TBC Paru
        if symptoms.get('batuk_lama') or symptoms.get('batuk_darah'):
            jenis_tbc.append("TBC Paru")
            
        # TBC Kelenjar
        if symptoms.get('benjolan_leher'):
            jenis_tbc.append("TBC Kelenjar (Limfadenitis)")
            
        # TBC Tulang/Sendi
        if symptoms.get('nyeri_tulang') or symptoms.get('bengkak_sendi'):
            jenis_tbc.append("TBC Tulang/Sendi")
            
        # TBC Selaput Otak
        if symptoms.get('sakit_kepala'):
            jenis_tbc.append("Kemungkinan TBC Selaput Otak (Meningitis TB)")
            
        if not jenis_tbc:
            jenis_tbc.append("Tidak terdeteksi jenis spesifik")
        
        # Tentukan status
        if self.symptoms_score >= 10:
            status = "RISIKO TINGGI"
            urgency = "SEGERA"
        elif self.symptoms_score >= 6:
            status = "RISIKO SEDANG"
            urgency = "DALAM 1-2 HARI"
        elif self.symptoms_score >= 3:
            status = "RISIKO RENDAH"
            urgency = "DALAM 1 MINGGU"
        else:
            status = "RISIKO MINIMAL"
            urgency = "MONITOR GEJALA"
            
        print(f"\nNama: {self.current_user['nama']}")
        print(f"Skor Gejala & Risiko: {self.symptoms_score}")
        print(f"Status: {status}")
        print(f"Kemungkinan Jenis TBC: {', '.join(jenis_tbc)}")
        
        # Rekomendasi
        print("\n💡 REKOMENDASI:")
        print("-" * 60)
        
        recommendations = []
        
        if self.symptoms_score >= 6:
            recommendations.append(f"🚨 {urgency} periksakan diri ke Puskesmas/Rumah Sakit")
            recommendations.append("📋 Minta pemeriksaan dahak (BTA) atau tes cepat TB (TCM/Xpert)")
            recommendations.append("🩻 Mungkin perlu foto rontgen dada")
            
        if "TBC Paru" in jenis_tbc:
            recommendations.append("😷 Gunakan masker saat batuk untuk mencegah penularan")
            recommendations.append("🏠 Tingkatkan ventilasi udara di rumah")
            
        if "TBC Kelenjar" in jenis_tbc:
            recommendations.append("🔬 Perlu pemeriksaan FNAB (Fine Needle Aspiration Biopsy)")
            
        if "TBC Tulang/Sendi" in jenis_tbc:
            recommendations.append("🏥 Konsultasi ke dokter ortopedi dan spesialis paru")
            
        if "Meningitis" in ' '.join(jenis_tbc):
            recommendations.append("🚑 SEGERA KE IGD - Kondisi ini darurat medis!")
            
        recommendations.extend([
            "🍎 Konsumsi makanan bergizi tinggi protein",
            "💊 JANGAN mengobati sendiri - TBC memerlukan pengobatan khusus 6-9 bulan",
            "👨‍👩‍👧 Informasikan ke keluarga untuk pemeriksaan kontak",
            "📞 Hubungi hotline TBC Kemenkes: 0812-9992-8400"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
            
        print("\n" + "=" * 60)
        print("⚕️  Program pengobatan TBC GRATIS di seluruh Puskesmas Indonesia")
        print("=" * 60)
        
        return {
            'skor': self.symptoms_score,
            'status': status,
            'jenis_tbc': ', '.join(jenis_tbc),
            'rekomendasi': '\n'.join(recommendations)
        }
        
    def save_consultation(self, symptoms, risk_factors, result):
        """Simpan hasil konsultasi ke database"""
        self.cursor.execute('''
            INSERT INTO consultations 
            (user_id, gejala, faktor_risiko, skor_total, jenis_tbc, status_deteksi, rekomendasi)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.current_user['id'],
            json.dumps(symptoms),
            json.dumps(risk_factors),
            result['skor'],
            result['jenis_tbc'],
            result['status'],
            result['rekomendasi']
        ))
        self.conn.commit()
        
    def show_statistics(self):
        """Tampilkan statistik deteksi TBC"""
        print("\n\n📈 STATISTIK DETEKSI TBC")
        print("=" * 60)
        
        # Total konsultasi
        self.cursor.execute('SELECT COUNT(*) FROM consultations')
        total = self.cursor.fetchone()[0]
        
        # Berdasarkan status
        self.cursor.execute('''
            SELECT status_deteksi, COUNT(*) 
            FROM consultations 
            GROUP BY status_deteksi
        ''')
        status_stats = self.cursor.fetchall()
        
        # Berdasarkan jenis TBC
        self.cursor.execute('''
            SELECT jenis_tbc, COUNT(*) 
            FROM consultations 
            WHERE jenis_tbc != 'Tidak terdeteksi jenis spesifik'
            GROUP BY jenis_tbc
            ORDER BY COUNT(*) DESC
            LIMIT 5
        ''')
        jenis_stats = self.cursor.fetchall()
        
        # Berdasarkan lokasi
        self.cursor.execute('''
            SELECT u.lokasi, COUNT(*) as jumlah
            FROM consultations c
            JOIN users u ON c.user_id = u.id
            GROUP BY u.lokasi
            ORDER BY jumlah DESC
            LIMIT 5
        ''')
        lokasi_stats = self.cursor.fetchall()
        
        print(f"\nTotal Konsultasi: {total}")
        
        print("\n--- Berdasarkan Status Risiko ---")
        for status, count in status_stats:
            print(f"{status}: {count} orang")
            
        print("\n--- Berdasarkan Jenis TBC ---")
        for jenis, count in jenis_stats:
            print(f"{jenis}: {count} kasus")
            
        print("\n--- Berdasarkan Lokasi (Top 5) ---")
        for lokasi, count in lokasi_stats:
            print(f"{lokasi}: {count} orang")
            
        print("\n" + "=" * 60)
        
    def close(self):
        """Tutup koneksi database"""
        self.conn.close()


# Main program
if __name__ == "__main__":
    chatbot = TBCChatbot()
    
    while True:
        chatbot.start_chat()
        
        print("\n\nApakah ada pengguna lain yang ingin konsultasi?")
        lanjut = input("Lanjut (y/t): ").lower()
        
        if lanjut != 'y':
            break
            
        # Reset untuk user baru
        chatbot.symptoms_score = 0
        chatbot.current_user = {}
    
    chatbot.close()
    print("\n✅ Terima kasih telah menggunakan sistem deteksi TBC!")
    print("Tetap jaga kesehatan dan segera konsultasi ke tenaga medis.\n")