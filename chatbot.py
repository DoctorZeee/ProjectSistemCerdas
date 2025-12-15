###
# TBC Detection System
# Chatbot
# Author: Firstname Lastname
# Date: 2023-10-01
###

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




with app.app_context():
    db.create_all()
    print("Database created successfully!")
    
    # Jika database kosong, generate data dummy otomatis
    if User.query.count() == 0:
        print("Database kosong, generating dummy data...")
        # generate_dummy_data()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)