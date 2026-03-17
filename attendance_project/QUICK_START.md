# Attendance Project - Quick Reference

## Launch the Application

```bash
python app.py
```

Open browser: **http://127.0.0.1:5000**

---

## Quick Commands

### Register a Student
Via Web UI: Home → Register Student → Enter Name → Capture Images

Via Command Line (if needed):
```bash
python capture.py "John Doe"
```

### Train Model
Via Web UI: Home → Train Model → Click Start Training

Via Command Line:
```bash
python train.py
```

### Take Attendance
Via Web UI: Home → Take Attendance → Click Start

Via Command Line:
```bash
python attendance.py
```

---

## File Locations

| What | Where | Status |
|------|-------|--------|
| Captured Images | `dataset/StudentName/*.jpg` | Auto-created |
| Trained Models | `models/svm.pkl`, `models/encoder.pkl` | Auto-created |
| Database | `attendance.db` | Auto-created |
| Web Pages | `templates/*.html` | Project files |
| Styling | `static/style.css` | Project files |
| Python Code | `*.py` files | Project files |

---

## Common Issues & Quick Fixes

| Problem | Solution |
|---------|----------|
| Camera won't open | Restart app, check camera permissions |
| Model training fails | Register at least 1 student first |
| Low recognition | Re-register with better images, retrain |
| Database error | Delete `attendance.db`, restart app |
| Port 5000 in use | Change in `app.py`: `port=5001` |

---

## What's New in This Version

✨ **v1.0 Improvements:**
- ✅ SQLite database instead of CSV
- ✅ Professional Bootstrap 5 UI
- ✅ Comprehensive error handling
- ✅ Model caching (FaceNet)
- ✅ Export to CSV feature
- ✅ Statistics dashboard
- ✅ Responsive mobile design
- ✅ Detailed logging

---

## Browser Compatibility

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

---

## Performance Notes

- **First Run**: FaceNet downloads (~170MB) - takes 2-5 minutes
- **Training Time**: ~1 second per student
- **Recognition Time**: ~500ms per face
- **Database**: Efficient with SQLite

---

## Need Help?

1. Check `README.md` for detailed documentation
2. Check error messages in console
3. Enable debug logging in `app.py`
4. Review template files for UI customization

---

Created: March 17, 2026
Version: 1.0
