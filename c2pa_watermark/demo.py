from C2PA_detector import detect_c2pa
from PIL import Image
import json

def make_cropped(input_path: str, output_path: str, ratio: float):
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    new_w, new_h = int(w * ratio), int(h * ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    img = img.crop((left, top, left + new_w, top + new_h))
    img.save(output_path, quality=95)
    print(f"[OK] cropped: {output_path}")

def make_resized(input_path: str, output_path: str, scale: float):
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)))
    img.save(output_path, quality=95)
    print(f"[OK] resized: {output_path}")

def make_converted(input_path: str, output_path: str, quality: int = 95) -> None:
    img = Image.open(input_path).convert("RGB")
    img.save(output_path, "JPEG", quality=quality)
    print(f"[OK] converted: {output_path}")

def classify_provenance_status(before_has_c2pa: bool, after_has_c2pa: bool) -> str:
    if before_has_c2pa and after_has_c2pa:
        return "c2pa_preserved"
    if before_has_c2pa and not after_has_c2pa:
        return "c2pa_lost_after_processing"
    if not before_has_c2pa and after_has_c2pa:
        return "c2pa_added_or_unexpected"
    return "no_c2pa_detected"

def record_to_summary(record):
    return {
        "path": record.path,
        "has_c2pa": record.has_c2pa,
        "active_manifest_label": record.active_manifest_label,
        "error": record.error,
    }

def run_case(case_name: str, original_path: str, processed_path: str):
    before = detect_c2pa(original_path)
    after = detect_c2pa(processed_path)

    status = classify_provenance_status(before.has_c2pa, after.has_c2pa)

    return {
        "case": case_name,
        "before": record_to_summary(before),
        "after": record_to_summary(after),
        "provenance_status": status,
    }

if __name__ == "__main__":
    gpt_c2pa = detect_c2pa("./demo.png")
    print(gpt_c2pa.to_json())

    results = []

    make_cropped("./demo.png", "./demo_cropped.png", 0.5)
    #print(detect_c2pa('./demo_cropped.png'))
    results.append(run_case("crop", "./demo.png", "./demo_cropped.png"))

    make_resized("./demo.png", "./demo_resized.png", 0.3)
    #print(detect_c2pa('./demo_resized.png'))
    results.append(run_case("resize", "./demo.png", "./demo_resized.png"))

    make_converted("./demo.png", "./demo_converted.jpeg")
    #print(detect_c2pa('./demo_converted.jpeg'))
    results.append(run_case("convert", "./demo.png", "./demo_converted.jpeg"))

    with open("report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)