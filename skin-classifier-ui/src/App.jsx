import { useState, useCallback } from "react";
import axios from "axios";

const API = "http://localhost:8000";

const CLASS_LABELS = {
  melanoma:   { label: "Melanoma",            color: "#dc2626" },
  nevus:      { label: "Melanocytic Nevus",    color: "#2563eb" },
  basal_cell: { label: "Basal Cell Carcinoma", color: "#d97706" },
  keratosis:  { label: "Benign Keratosis",     color: "#16a34a" },
};

export default function App() {
  const [file,    setFile]    = useState(null);
  const [preview, setPreview] = useState(null);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post(`${API}/predict`, form);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Something went wrong. Try another image.");
    } finally {
      setLoading(false);
    }
  };

  const topClass = result ? CLASS_LABELS[result.class] : null;

  return (
    <div style={{ maxWidth: 640, margin: "0 auto", padding: "2rem 1rem",
      fontFamily: "system-ui, sans-serif" }}>

      <h1 style={{ fontSize: 22, fontWeight: 600, margin: "0 0 4px" }}>
        Skin Condition Classifier
      </h1>
      <p style={{ color: "#6b7280", fontSize: 14, margin: "0 0 24px" }}>
        Upload a dermoscopy image to classify it. Research prototype — not for medical use.
      </p>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => document.getElementById("fileInput").click()}
        style={{
          border: "2px dashed #d1d5db", borderRadius: 12, padding: "2rem",
          textAlign: "center", cursor: "pointer", marginBottom: 12,
          background: preview ? "#f9fafb" : "white", minHeight: 160,
          display: "flex", alignItems: "center", justifyContent: "center"
        }}
      >
        {preview
          ? <img src={preview} alt="preview"
              style={{ maxHeight: 240, maxWidth: "100%", borderRadius: 8 }} />
          : <div>
              <p style={{ color: "#9ca3af", margin: 0 }}>
                Drop an image here, or click to upload
              </p>
              <p style={{ color: "#d1d5db", fontSize: 12, margin: "4px 0 0" }}>
                JPEG, PNG, WebP — max 10MB
              </p>
            </div>
        }
      </div>

      <input id="fileInput" type="file" accept="image/*"
        onChange={e => handleFile(e.target.files[0])}
        style={{ display: "none" }} />

      {/* Analyse button */}
      <button
        onClick={handlePredict}
        disabled={!file || loading}
        style={{
          width: "100%", padding: 12, borderRadius: 8, border: "none",
          background: file && !loading ? "#4f46e5" : "#e5e7eb",
          color: file && !loading ? "white" : "#9ca3af",
          fontSize: 15, fontWeight: 500,
          cursor: file && !loading ? "pointer" : "default",
          marginBottom: 24
        }}
      >
        {loading ? "Analysing…" : "Analyse image"}
      </button>

      {/* Error */}
      {error && (
        <div style={{ background: "#fef2f2", border: "1px solid #fecaca",
          borderRadius: 8, padding: "12px 16px", color: "#dc2626", marginBottom: 16 }}>
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div>
          {/* Main prediction card */}
          <div style={{ background: "#f9fafb", border: "1px solid #e5e7eb",
            borderRadius: 12, padding: "1.25rem", marginBottom: 16 }}>
            <p style={{ color: "#6b7280", fontSize: 13, margin: "0 0 4px" }}>
              Predicted condition
            </p>
            <p style={{ fontSize: 22, fontWeight: 600,
              color: topClass.color, margin: "0 0 4px" }}>
              {topClass.label}
            </p>
            <p style={{ fontSize: 14, color: "#6b7280", margin: 0 }}>
              Confidence: {result.confidence}%
            </p>
          </div>

          {/* Score bars */}
          <div style={{ marginBottom: 24 }}>
            <p style={{ fontSize: 13, fontWeight: 500, margin: "0 0 10px" }}>
              All class scores
            </p>
            {Object.entries(result.all_scores)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, score]) => (
                <div key={cls} style={{ marginBottom: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between",
                    fontSize: 13, marginBottom: 4 }}>
                    <span>{CLASS_LABELS[cls]?.label || cls}</span>
                    <span style={{ color: "#6b7280" }}>{score}%</span>
                  </div>
                  <div style={{ background: "#e5e7eb", borderRadius: 4, height: 6 }}>
                    <div style={{
                      width: `${score}%`, height: "100%", borderRadius: 4,
                      background: CLASS_LABELS[cls]?.color || "#6b7280",
                      transition: "width 0.6s ease"
                    }} />
                  </div>
                </div>
              ))}
          </div>

          {/* Grad-CAM */}
          <div style={{ marginBottom: 24 }}>
            <p style={{ fontSize: 13, fontWeight: 500, margin: "0 0 4px" }}>
              What the model focused on
            </p>
            <p style={{ fontSize: 12, color: "#9ca3af", margin: "0 0 8px" }}>
              Red areas had the highest influence on the prediction.
            </p>
            <img
              src={`data:image/jpeg;base64,${result.heatmap_base64}`}
              alt="Grad-CAM heatmap"
              style={{ width: "100%", borderRadius: 8 }}
            />
          </div>

          {/* Disclaimer */}
          <div style={{ background: "#fffbeb", border: "1px solid #fde68a",
            borderRadius: 8, padding: "12px 16px", fontSize: 13, color: "#92400e" }}>
            ⚠️ This is a research prototype built for a student portfolio.
            It is not a medical device and must not be used for diagnosis.
            Always consult a qualified dermatologist.
          </div>
        </div>
      )}
    </div>
  );
}