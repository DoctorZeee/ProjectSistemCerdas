from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import os
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

app = Flask(__name__)

# Konfigurasi Database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'tbc_detection.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'tbc-detection-secret-key-2024'

db = SQLAlchemy(app)

# Model Database
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    usia = db.Column(db.Integer)
    jenis_kelamin = db.Column(db.String(20))
    lokasi = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    consultations = db.relationship('Consultation', backref='user', lazy=True)

class Consultation(db.Model):
    __tablename__ = 'consultations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    gejala = db.Column(db.Text)
    faktor_risiko = db.Column(db.Text)
    skor_total = db.Column(db.Integer)
    jenis_tbc = db.Column(db.String(200))
    status_deteksi = db.Column(db.String(50))
    rekomendasi = db.Column(db.Text)
    cerita = db.Column(db.Text)
    ground_truth = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Pertanyaan Gejala TBC
GEJALA_QUESTIONS = [
    ("batuk_lama", "Apakah Anda mengalami batuk lebih dari 2-3 minggu?", 3, "paru"),
    ("batuk_darah", "Apakah batuk disertai darah?", 4, "paru"),
    ("demam", "Apakah mengalami demam (terutama sore/malam)?", 2, "paru"),
    ("keringat_malam", "Apakah berkeringat banyak di malam hari?", 2, "paru"),
    ("penurunan_bb", "Apakah mengalami penurunan berat badan tanpa sebab jelas?", 3, "paru"),
    ("nafsu_makan", "Apakah nafsu makan menurun?", 1, "paru"),
    ("sesak_napas", "Apakah mengalami sesak napas?", 2, "paru"),
    ("nyeri_dada", "Apakah merasakan nyeri dada?", 2, "paru"),
    ("lemah_lesu", "Apakah merasa lemah dan lesu berkepanjangan?", 1, "paru"),
    ("benjolan_leher", "Apakah ada benjolan di leher/ketiak?", 2, "kelenjar"),
    ("nyeri_tulang", "Apakah ada nyeri tulang/sendi yang persisten?", 2, "tulang"),
    ("bengkak_sendi", "Apakah ada pembengkakan pada sendi?", 2, "tulang"),
    ("sakit_kepala", "Apakah mengalami sakit kepala hebat/terus-menerus?", 2, "selaput_otak"),
]

RISK_QUESTIONS = [
    ("kontak_tbc", "Apakah pernah kontak dekat dengan penderita TBC?", 3),
    ("riwayat_tbc", "Apakah pernah menderita TBC sebelumnya?", 3),
    ("hiv", "Apakah memiliki HIV/AIDS atau gangguan sistem imun?", 4),
    ("diabetes", "Apakah memiliki diabetes?", 2),
    ("merokok", "Apakah Anda perokok aktif?", 1),
    ("lingkungan_padat", "Apakah tinggal di lingkungan padat/kurang ventilasi?", 1),
    ("gizi_buruk", "Apakah mengalami kekurangan gizi?", 2),
]

STORY_KEYWORDS = {
    "batuk darah": 4, "batuk berdarah": 4, "dahak berdarah": 4,
    "batuk lama": 3, "batuk terus": 3, "batuk berkepanjangan": 3,
    "demam": 2, "panas": 2, "keringat malam": 2, "berkeringat malam": 2,
    "berat badan turun": 3, "kurus": 2, "nafsu makan hilang": 1,
    "sesak": 2, "susah napas": 2, "napas pendek": 2,
    "lemas": 1, "lemah": 1, "capek": 1, "lelah": 1, "lesu": 1,
    "benjolan": 2, "nyeri tulang": 2, "sakit kepala": 2,
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/deteksi")
def deteksi():
    return render_template("deteksi.html")

@app.route("/statistik")
def statistik():
    try:
        # Statistik utama
        total_consultations = Consultation.query.count()
        status_tinggi = Consultation.query.filter_by(status_deteksi="RISIKO TINGGI").count()
        status_sedang = Consultation.query.filter_by(status_deteksi="RISIKO SEDANG").count()
        status_rendah = Consultation.query.filter_by(status_deteksi="RISIKO RENDAH").count()
        status_minimal = Consultation.query.filter_by(status_deteksi="RISIKO MINIMAL").count()
        
        avg_score_result = db.session.query(db.func.avg(Consultation.skor_total)).scalar()
        avg_score = round(float(avg_score_result), 2) if avg_score_result else 0.0
        
        laki_laki = User.query.filter_by(jenis_kelamin="Laki-laki").count()
        perempuan = User.query.filter_by(jenis_kelamin="Perempuan").count()
        
        # Distribusi umur
        all_users = User.query.all()
        umur_0_17 = sum(1 for u in all_users if u.usia and u.usia < 18)
        umur_18_29 = sum(1 for u in all_users if u.usia and 18 <= u.usia < 30)
        umur_30_44 = sum(1 for u in all_users if u.usia and 30 <= u.usia < 45)
        umur_45_59 = sum(1 for u in all_users if u.usia and 45 <= u.usia < 60)
        umur_60_plus = sum(1 for u in all_users if u.usia and u.usia >= 60)
        
        # Top 5 Lokasi - FIXED QUERY
        lokasi_results = db.session.query(
            User.lokasi, 
            db.func.count(Consultation.id).label('jumlah')
        ).join(Consultation).filter(
            User.lokasi.isnot(None),
            User.lokasi != ''
        ).group_by(User.lokasi).order_by(db.desc('jumlah')).limit(5).all()
        
        lokasi_stats = []
        for lokasi, jumlah in lokasi_results:
            lokasi_stats.append({
                "lokasi": lokasi if lokasi else "Tidak diketahui",
                "jumlah": jumlah
            })
        
        # Jika tidak ada data lokasi
        if not lokasi_stats:
            lokasi_stats = [{"lokasi": "Data belum tersedia", "jumlah": 0}]
        
        # Jenis TBC yang terdeteksi - FIXED QUERY
        jenis_tbc_results = db.session.query(
            Consultation.jenis_tbc,
            db.func.count(Consultation.id).label('jumlah')
        ).filter(
            Consultation.jenis_tbc.isnot(None),
            Consultation.jenis_tbc != ""
        ).group_by(Consultation.jenis_tbc).order_by(db.desc('jumlah')).limit(5).all()
        
        jenis_tbc_stats = []
        for jenis, jumlah in jenis_tbc_results:
            jenis_tbc_stats.append({
                "jenis": jenis if jenis else "Tidak diketahui",
                "jumlah": jumlah
            })
        
        # Jika tidak ada data jenis TBC
        if not jenis_tbc_stats:
            jenis_tbc_stats = [{"jenis": "Data belum tersedia", "jumlah": 0}]
        
        # Data konsultasi terbaru untuk tabel
        recent_consultations = db.session.query(
            Consultation, User
        ).join(User).order_by(Consultation.created_at.desc()).limit(10).all()
        
        # Hitung metrik evaluasi
        evaluation_metrics = calculate_evaluation_metrics()
        
        # Siapkan stats untuk template
        stats = {
            "total": total_consultations,
            "tinggi": status_tinggi,
            "sedang": status_sedang,
            "rendah": status_rendah,
            "minimal": status_minimal,
            "avg_score": avg_score,
            "laki_laki": laki_laki,
            "perempuan": perempuan,
            "umur_0_17": umur_0_17,
            "umur_18_29": umur_18_29,
            "umur_30_44": umur_30_44,
            "umur_45_59": umur_45_59,
            "umur_60_plus": umur_60_plus,
            "lokasi_stats": lokasi_stats,
            "jenis_tbc_stats": jenis_tbc_stats,
            "recent_consultations": recent_consultations,
            "evaluation": evaluation_metrics
        }
        
        # Debug logging
        print(f"Lokasi stats: {lokasi_stats}")
        print(f"Jenis TBC stats: {jenis_tbc_stats}")
        
        return render_template("statistik.html", stats=stats)
    
    except Exception as e:
        print(f"Error di route statistik: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return data default
        stats = {
            "total": 0, "tinggi": 0, "sedang": 0, "rendah": 0, "minimal": 0,
            "avg_score": 0.0, "laki_laki": 0, "perempuan": 0,
            "umur_0_17": 0, "umur_18_29": 0, "umur_30_44": 0, 
            "umur_45_59": 0, "umur_60_plus": 0,
            "lokasi_stats": [{"lokasi": "Data belum tersedia", "jumlah": 0}],
            "jenis_tbc_stats": [{"jenis": "Data belum tersedia", "jumlah": 0}],
            "recent_consultations": [],
            "evaluation": {
                "has_data": False,
                "message": "Error loading data"
            }
        }
        return render_template("statistik.html", stats=stats)

def calculate_evaluation_metrics():
    """Hitung metrik evaluasi model"""
    try:
        consultations = Consultation.query.filter(
            Consultation.ground_truth.isnot(None),
            Consultation.ground_truth != ''
        ).all()
        
        if len(consultations) < 5:
            return {
                "has_data": False,
                "message": "Tidak cukup data untuk evaluasi (minimal 5 data dengan ground truth)"
            }
        
        y_pred = []
        y_true = []
        
        for c in consultations:
            y_pred.append(c.status_deteksi)
            y_true.append(c.ground_truth)
        
        labels = ["RISIKO MINIMAL", "RISIKO RENDAH", "RISIKO SEDANG", "RISIKO TINGGI"]
        
        # Hitung metrics
        f1_weighted = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Hitung accuracy manual
        correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
        accuracy = correct / len(y_true) if len(y_true) > 0 else 0
        
        # Siapkan F1 per class dalam dictionary
        f1_per_class_dict = {}
        for i, label in enumerate(labels):
            f1_per_class_dict[label] = round(f1_per_class[i] * 100, 2) if i < len(f1_per_class) else 0
        
        return {
            "has_data": True,
            "total_evaluated": len(consultations),
            "accuracy": round(accuracy * 100, 2),
            "f1_score": round(f1_weighted * 100, 2),
            "precision": round(precision_weighted * 100, 2),
            "recall": round(recall_weighted * 100, 2),
            "f1_per_class": f1_per_class_dict,
            "confusion_matrix": cm.tolist(),
            "labels": labels
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            "has_data": False,
            "message": f"Error: {str(e)}"
        }

@app.route("/api/get_questions", methods=["POST"])
def get_questions():
    try:
        data = request.json
        question_type = data.get("type", "gejala")
        
        if question_type == "gejala":
            questions = [
                {"key": key, "text": text, "score": score, "category": cat}
                for key, text, score, cat in GEJALA_QUESTIONS
            ]
        else:
            questions = [
                {"key": key, "text": text, "score": score}
                for key, text, score in RISK_QUESTIONS
            ]
        
        return jsonify({"questions": questions})
    except Exception as e:
        print(f"Error in get_questions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if not request.json:
            return jsonify({
                "success": False, 
                "error": "No JSON data received"
            }), 400
        
        data = request.json
        
        nama = data.get("nama", "Anonim")
        usia = int(data.get("usia", 0)) if str(data.get("usia", "0")).isdigit() else 0
        jenis_kelamin = data.get("jenis_kelamin", "Tidak disebutkan")
        lokasi = data.get("lokasi", "Tidak disebutkan")
        gejala = data.get("gejala", {})
        faktor_risiko = data.get("faktor_risiko", {})
        cerita = str(data.get("cerita", "")).lower()
        
        total_score = 0
        
        for key, text, score, cat in GEJALA_QUESTIONS:
            if gejala.get(key) == True:
                total_score += score
        
        for key, text, score in RISK_QUESTIONS:
            if faktor_risiko.get(key) == True:
                total_score += score
        
        detected_keywords = []
        if cerita:
            for keyword, score in STORY_KEYWORDS.items():
                if keyword in cerita:
                    total_score += score
                    detected_keywords.append({"keyword": keyword, "score": score})
        
        jenis_tbc = []
        if gejala.get('batuk_lama') or gejala.get('batuk_darah') or gejala.get('demam'):
            jenis_tbc.append("TBC Paru")
        if gejala.get('benjolan_leher'):
            jenis_tbc.append("TBC Kelenjar (Limfadenitis)")
        if gejala.get('nyeri_tulang') or gejala.get('bengkak_sendi'):
            jenis_tbc.append("TBC Tulang/Sendi")
        if gejala.get('sakit_kepala'):
            jenis_tbc.append("Kemungkinan TBC Selaput Otak (Meningitis TB)")
        if not jenis_tbc:
            jenis_tbc.append("Tidak terdeteksi jenis spesifik")
        
        if total_score >= 10:
            status = "RISIKO TINGGI"
            urgency = "SEGERA"
            color = "danger"
        elif total_score >= 6:
            status = "RISIKO SEDANG"
            urgency = "DALAM 1-2 HARI"
            color = "warning"
        elif total_score >= 3:
            status = "RISIKO RENDAH"
            urgency = "DALAM 1 MINGGU"
            color = "info"
        else:
            status = "RISIKO MINIMAL"
            urgency = "MONITOR GEJALA"
            color = "success"
        
        recommendations = generate_recommendations(total_score, jenis_tbc, urgency)
        
        # Simpan ke database
        new_user = User(
            nama=nama,
            usia=usia,
            jenis_kelamin=jenis_kelamin,
            lokasi=lokasi
        )
        db.session.add(new_user)
        db.session.flush()
        
        new_consultation = Consultation(
            user_id=new_user.id,
            gejala=json.dumps(gejala),
            faktor_risiko=json.dumps(faktor_risiko),
            skor_total=total_score,
            jenis_tbc=", ".join(jenis_tbc),
            status_deteksi=status,
            rekomendasi="\n".join(recommendations),
            cerita=cerita,
            ground_truth=None
        )
        db.session.add(new_consultation)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "result": {
                "nama": nama,
                "skor": total_score,
                "status": status,
                "color": color,
                "jenis_tbc": jenis_tbc,
                "rekomendasi": recommendations,
                "detected_keywords": detected_keywords,
                "urgency": urgency,
                "consultation_id": new_consultation.id
            }
        })
    
    except Exception as e:
        db.session.rollback()
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error di analyze: {str(e)}")
        print(f"Traceback: {error_detail}")
        
        return jsonify({
            "success": False, 
            "error": str(e),
            "detail": error_detail
        }), 500

def generate_recommendations(score, jenis_tbc, urgency):
    recommendations = []
    
    if score >= 6:
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
        "📞 Hubungi hotline TBC Kemenkes: 0812-9992-8400",
        "⚕️ Program pengobatan TBC GRATIS di seluruh Puskesmas Indonesia"
    ])
    
    return recommendations

@app.route("/api/dev/generate_realistic_data")
def generate_realistic_data():
    """Generate data realistik dengan semua kategori risiko"""
    try:
        from datetime import datetime, timedelta
        import random
        
        # Hapus data lama jika ada
        Consultation.query.delete()
        User.query.delete()
        db.session.commit()
        
        print("Generating realistic data with all risk categories...")
        
        # Distribusi kategori risiko yang kita inginkan
        risk_distribution = {
            "RISIKO TINGGI": 25,      # 25 data
            "RISIKO SEDANG": 25,      # 25 data  
            "RISIKO RENDAH": 25,      # 25 data
            "RISIKO MINIMAL": 25      # 25 data
        }
        
        total_data = 100
        data_per_category = 25
        
        lokasi_list = ["Jakarta", "Bandung", "Surabaya", "Yogyakarta", "Medan"]
        jenis_kelamin_list = ["Laki-laki", "Perempuan"]
        
        # Generate untuk setiap kategori
        for category_idx, (category_name, count) in enumerate(risk_distribution.items()):
            print(f"Generating {count} data for {category_name}...")
            
            for i in range(count):
                # ID unik berdasarkan kategori
                data_id = category_idx * data_per_category + i + 1
                
                # Tentukan skor berdasarkan kategori
                if category_name == "RISIKO TINGGI":
                    # Skor 10-15
                    target_score = random.randint(10, 15)
                    usia = random.randint(40, 75)  # Umur lebih tua
                    jenis_tbc = "TBC Paru"
                    
                elif category_name == "RISIKO SEDANG":
                    # Skor 6-9
                    target_score = random.randint(6, 9)
                    usia = random.randint(25, 60)
                    jenis_tbc = random.choice(["TBC Paru", "TBC Kelenjar (Limfadenitis)"])
                    
                elif category_name == "RISIKO RENDAH":
                    # Skor 3-5
                    target_score = random.randint(3, 5)
                    usia = random.randint(18, 50)
                    jenis_tbc = random.choice(["TBC Tulang/Sendi", "Tidak terdeteksi jenis spesifik"])
                    
                else:  # RISIKO MINIMAL
                    # Skor 0-2
                    target_score = random.randint(0, 2)
                    usia = random.randint(15, 40)
                    jenis_tbc = "Tidak terdeteksi jenis spesifik"
                
                # Untuk kategori RENDAH dan MINIMAL, bisa termasuk pasien diabetes
                if category_name in ["RISIKO RENDAH", "RISIKO MINIMAL"]:
                    # 30% dari data rendah/minimal adalah pasien diabetes
                    is_diabetic = random.random() < 0.3
                else:
                    is_diabetic = random.random() < 0.1  # 10% untuk tinggi/sedang
                
                # Buat gejala dan faktor risiko berdasarkan target skor
                gejala_dict = {}
                faktor_dict = {}
                
                # GEJALA untuk mencapai target skor
                gejala_available = [
                    ("batuk_lama", 3),
                    ("demam", 2),
                    ("keringat_malam", 2),
                    ("penurunan_bb", 3),
                    ("lemah_lesu", 1),
                    ("sesak_napas", 2),
                    ("nyeri_dada", 2),
                    ("batuk_darah", 4),
                    ("nafsu_makan", 1)
                ]
                
                current_score = 0
                
                # Tambahkan gejala sampai mendekati target
                for gejala, score in sorted(gejala_available, key=lambda x: x[1], reverse=True):
                    if current_score >= target_score:
                        break
                    
                    # Probabilitas gejala berdasarkan kategori
                    if category_name == "RISIKO TINGGI":
                        prob = 0.8
                    elif category_name == "RISIKO SEDANG":
                        prob = 0.6
                    elif category_name == "RISIKO RENDAH":
                        prob = 0.4
                    else:
                        prob = 0.2
                    
                    if random.random() < prob and current_score + score <= target_score + 2:
                        gejala_dict[gejala] = True
                        current_score += score
                    else:
                        gejala_dict[gejala] = False
                
                # FAKTOR RISIKO
                if is_diabetic:
                    faktor_dict["diabetes"] = True
                    current_score += 2  # diabetes = 2 poin
                    
                    # Untuk pasien diabetes minimal, tambah gejala ringan
                    if category_name == "RISIKO MINIMAL" and current_score < 3:
                        gejala_dict["lemah_lesu"] = True
                        current_score += 1
                
                # Faktor risiko lainnya
                if random.random() < 0.3:
                    faktor_dict["merokok"] = True
                    current_score += 1
                
                if random.random() < 0.2:
                    faktor_dict["kontak_tbc"] = True
                    current_score += 3
                
                if random.random() < 0.15:
                    faktor_dict["gizi_buruk"] = True
                    current_score += 2
                
                # Normalisasi skor jika melebihi target
                if current_score > target_score + 2:
                    # Kurangi beberapa gejala ringan
                    if gejala_dict.get("lemah_lesu"):
                        gejala_dict["lemah_lesu"] = False
                        current_score -= 1
                    if gejala_dict.get("nafsu_makan"):
                        gejala_dict["nafsu_makan"] = False
                        current_score -= 1
                
                # Tentukan status akhir berdasarkan skor
                if current_score >= 10:
                    final_status = "RISIKO TINGGI"
                elif current_score >= 6:
                    final_status = "RISIKO SEDANG"
                elif current_score >= 3:
                    final_status = "RISIKO RENDAH"
                else:
                    final_status = "RISIKO MINIMAL"
                
                # Buat user
                user = User(
                    nama=f"Pasien {category_name[:1]}{data_id:03d}",
                    usia=usia,
                    jenis_kelamin=random.choice(jenis_kelamin_list),
                    lokasi=random.choice(lokasi_list),
                    created_at=datetime.utcnow() - timedelta(days=random.randint(1, 365))
                )
                db.session.add(user)
                db.session.flush()
                
                # Buat cerita
                if is_diabetic:
                    cerita = f"Pasien {usia} tahun dengan diabetes. "
                else:
                    cerita = f"Pasien {usia} tahun. "
                
                if gejala_dict.get("batuk_lama"):
                    cerita += "Mengalami batuk berkepanjangan. "
                if gejala_dict.get("demam"):
                    cerita += "Demam terutama sore/malam. "
                
                # Rekomendasi
                if final_status == "RISIKO TINGGI":
                    rekomendasi = "🚨 SEGERA ke fasilitas kesehatan. Tes dahak dan rontgen dada."
                elif final_status == "RISIKO SEDANG":
                    rekomendasi = "📅 Dalam 1-2 hari periksa ke puskesmas."
                elif final_status == "RISIKO RENDAH":
                    rekomendasi = "📋 Konsultasi dokter dalam 1 minggu."
                else:
                    rekomendasi = "👁️ Monitor gejala, kontrol rutin jika ada keluhan."
                
                if is_diabetic:
                    rekomendasi += "\n💉 Kontrol gula darah rutin penting untuk pasien diabetes."
                
                # Buat konsultasi
                consultation = Consultation(
                    user_id=user.id,
                    gejala=json.dumps(gejala_dict),
                    faktor_risiko=json.dumps(faktor_dict),
                    skor_total=current_score,
                    jenis_tbc=jenis_tbc,
                    status_deteksi=final_status,
                    rekomendasi=rekomendasi,
                    cerita=cerita,
                    ground_truth=final_status,
                    created_at=user.created_at
                )
                db.session.add(consultation)
            
            # Commit per kategori
            db.session.flush()
        
        db.session.commit()
        
        # Hitung statistik akhir
        stats = {
            "total": Consultation.query.count(),
            "tinggi": Consultation.query.filter_by(status_deteksi="RISIKO TINGGI").count(),
            "sedang": Consultation.query.filter_by(status_deteksi="RISIKO SEDANG").count(),
            "rendah": Consultation.query.filter_by(status_deteksi="RISIKO RENDAH").count(),
            "minimal": Consultation.query.filter_by(status_deteksi="RISIKO MINIMAL").count(),
            "diabetes": Consultation.query.filter(Consultation.faktor_risiko.like('%diabetes%')).count()
        }
        
        return jsonify({
            "success": True,
            "message": f"Generated {total_data} realistic data with balanced distribution",
            "stats": stats,
            "note": "Semua kategori risiko terwakili termasuk pasien diabetes di kategori rendah/minimal"
        })
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/debug_data")
def debug_data():
    """Debug endpoint untuk melihat data di database"""
    try:
        total_users = User.query.count()
        total_consultations = Consultation.query.count()
        
        lokasi_stats = db.session.query(
            User.lokasi, 
            db.func.count(Consultation.id).label('jumlah')
        ).join(Consultation).group_by(User.lokasi).all()
        
        jenis_tbc_stats = db.session.query(
            Consultation.jenis_tbc,
            db.func.count(Consultation.id).label('jumlah')
        ).group_by(Consultation.jenis_tbc).all()
        
        return jsonify({
            "total_users": total_users,
            "total_consultations": total_consultations,
            "lokasi_stats": [{"lokasi": l[0], "jumlah": l[1]} for l in lokasi_stats],
            "jenis_tbc_stats": [{"jenis": j[0], "jumlah": j[1]} for j in jenis_tbc_stats],
            "sample_users": [
                {"id": u.id, "nama": u.nama, "lokasi": u.lokasi}
                for u in User.query.limit(5).all()
            ],
            "sample_consultations": [
                {"id": c.id, "jenis_tbc": c.jenis_tbc, "status": c.status_deteksi}
                for c in Consultation.query.limit(5).all()
            ]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

with app.app_context():
    db.create_all()
    print("Database created successfully!")
    
    # Jika database kosong, generate data dummy otomatis
    if User.query.count() == 0:
        print("Database kosong, generating dummy data...")
        generate_dummy_data()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)