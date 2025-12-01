from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import os

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

# Pertanyaan Faktor Risiko
RISK_QUESTIONS = [
    ("kontak_tbc", "Apakah pernah kontak dekat dengan penderita TBC?", 3),
    ("riwayat_tbc", "Apakah pernah menderita TBC sebelumnya?", 3),
    ("hiv", "Apakah memiliki HIV/AIDS atau gangguan sistem imun?", 4),
    ("diabetes", "Apakah memiliki diabetes?", 2),
    ("merokok", "Apakah Anda perokok aktif?", 1),
    ("lingkungan_padat", "Apakah tinggal di lingkungan padat/kurang ventilasi?", 1),
    ("gizi_buruk", "Apakah mengalami kekurangan gizi?", 2),
]

# Kata kunci untuk analisis cerita
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

# @app.route("/deteksi")
# def deteksi():
#     return render_template("deteksi.html")

@app.route("/statistik")
def statistik():
    try:
        # Total konsultasi
        total_consultations = Consultation.query.count()
        
        # Berdasarkan status
        status_tinggi = Consultation.query.filter_by(status_deteksi="RISIKO TINGGI").count()
        status_sedang = Consultation.query.filter_by(status_deteksi="RISIKO SEDANG").count()
        status_rendah = Consultation.query.filter_by(status_deteksi="RISIKO RENDAH").count()
        status_minimal = Consultation.query.filter_by(status_deteksi="RISIKO MINIMAL").count()
        
        # Rata-rata skor
        avg_score_result = db.session.query(db.func.avg(Consultation.skor_total)).scalar()
        avg_score = round(float(avg_score_result), 2) if avg_score_result else 0.0
        
        # Berdasarkan gender
        laki_laki = User.query.filter_by(jenis_kelamin="Laki-laki").count()
        perempuan = User.query.filter_by(jenis_kelamin="Perempuan").count()
        
        # Berdasarkan kelompok umur
        all_users = User.query.all()
        umur_0_17 = sum(1 for u in all_users if u.usia and u.usia < 18)
        umur_18_29 = sum(1 for u in all_users if u.usia and 18 <= u.usia < 30)
        umur_30_44 = sum(1 for u in all_users if u.usia and 30 <= u.usia < 45)
        umur_45_59 = sum(1 for u in all_users if u.usia and 45 <= u.usia < 60)
        umur_60_plus = sum(1 for u in all_users if u.usia and u.usia >= 60)
        
        # Berdasarkan lokasi (Top 5)
        lokasi_stats = db.session.query(
            User.lokasi, 
            db.func.count(Consultation.id).label('jumlah')
        ).join(Consultation).filter(
            User.lokasi != None,
            User.lokasi != '',
            User.lokasi != 'Tidak disebutkan'
        ).group_by(User.lokasi).order_by(db.desc('jumlah')).limit(5).all()
        
        # Berdasarkan jenis TBC
        jenis_tbc_stats = db.session.query(
            Consultation.jenis_tbc,
            db.func.count(Consultation.id).label('jumlah')
        ).filter(
            Consultation.jenis_tbc != None,
            Consultation.jenis_tbc != "Tidak terdeteksi jenis spesifik",
            Consultation.jenis_tbc != ""
        ).group_by(Consultation.jenis_tbc).order_by(db.desc('jumlah')).limit(5).all()
        
        # Data konsultasi terbaru (10 terakhir)
        recent_consultations = db.session.query(
            Consultation, User
        ).join(User).order_by(Consultation.created_at.desc()).limit(10).all()
        
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
            "lokasi_stats": lokasi_stats if lokasi_stats else [],
            "jenis_tbc_stats": jenis_tbc_stats if jenis_tbc_stats else [],
            "recent_consultations": recent_consultations if recent_consultations else []
        }
        
        return render_template("statistik.html", stats=stats)
    
    except Exception as e:
        print(f"Error di route statistik: {str(e)}")
        # Return data kosong jika error
        stats = {
            "total": 0,
            "tinggi": 0,
            "sedang": 0,
            "rendah": 0,
            "minimal": 0,
            "avg_score": 0.0,
            "laki_laki": 0,
            "perempuan": 0,
            "umur_0_17": 0,
            "umur_18_29": 0,
            "umur_30_44": 0,
            "umur_45_59": 0,
            "umur_60_plus": 0,
            "lokasi_stats": [],
            "jenis_tbc_stats": [],
            "recent_consultations": []
        }
        return render_template("statistik.html", stats=stats)

@app.route("/api/get_questions", methods=["POST"])
def get_questions():
    """Return semua pertanyaan gejala dan faktor risiko"""
    try:
        data = request.json
        question_type = data.get("type", "gejala")
        
        if question_type == "gejala":
            questions = [
                {"key": key, "text": text, "score": score, "category": cat}
                for key, text, score, cat in GEJALA_QUESTIONS
            ]
        else:  # risiko
            questions = [
                {"key": key, "text": text, "score": score}
                for key, text, score in RISK_QUESTIONS
            ]
        
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analisis lengkap hasil deteksi"""
    try:
        data = request.json
        
        # Data user
        nama = data.get("nama", "Anonim")
        usia = int(data.get("usia", 0))
        jenis_kelamin = data.get("jenis_kelamin", "Tidak disebutkan")
        lokasi = data.get("lokasi", "Tidak disebutkan")
        
        # Data jawaban
        gejala = data.get("gejala", {})
        faktor_risiko = data.get("faktor_risiko", {})
        cerita = data.get("cerita", "").lower()
        
        # Hitung skor dari jawaban
        total_score = 0
        
        # Skor dari gejala
        for key, text, score, cat in GEJALA_QUESTIONS:
            if gejala.get(key) == True:
                total_score += score
        
        # Skor dari faktor risiko
        for key, text, score in RISK_QUESTIONS:
            if faktor_risiko.get(key) == True:
                total_score += score
        
        # Analisis kata kunci dari cerita
        detected_keywords = []
        for keyword, score in STORY_KEYWORDS.items():
            if keyword in cerita:
                total_score += score
                detected_keywords.append({"keyword": keyword, "score": score})
        
        # Tentukan jenis TBC
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
        
        # Tentukan status dan rekomendasi
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
        
        # Generate rekomendasi
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
        
        # Simpan consultation
        new_consultation = Consultation(
            user_id=new_user.id,
            gejala=json.dumps(gejala),
            faktor_risiko=json.dumps(faktor_risiko),
            skor_total=total_score,
            jenis_tbc=", ".join(jenis_tbc),
            status_deteksi=status,
            rekomendasi="\n".join(recommendations),
            cerita=cerita
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
                "urgency": urgency
            }
        })
    
    except Exception as e:
        db.session.rollback()
        print(f"Error di analyze: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

def generate_recommendations(score, jenis_tbc, urgency):
    """Generate rekomendasi berdasarkan hasil analisis"""
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

@app.route("/api/stats_chart", methods=["GET"])
def stats_chart():
    """API untuk chart data"""
    try:
        status_data = {
            "labels": ["Risiko Tinggi", "Risiko Sedang", "Risiko Rendah", "Risiko Minimal"],
            "values": [
                Consultation.query.filter_by(status_deteksi="RISIKO TINGGI").count(),
                Consultation.query.filter_by(status_deteksi="RISIKO SEDANG").count(),
                Consultation.query.filter_by(status_deteksi="RISIKO RENDAH").count(),
                Consultation.query.filter_by(status_deteksi="RISIKO MINIMAL").count(),
            ]
        }
        return jsonify(status_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Inisialisasi database
with app.app_context():
    db.create_all()
    print("Database created successfully!")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)