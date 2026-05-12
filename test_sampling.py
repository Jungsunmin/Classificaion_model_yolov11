import os
import random
from ultralytics import YOLO
from pathlib import Path

def run_random_test(num_samples=100):
    # 1. 모델 로드
    model_path = 'runs/classify/flea_market_cls/test_run_5_classes-2/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"에러: 모델 파일을 찾을 수 없습니다: {model_path}")
        print("학습을 먼저 완료하거나 모델 파일 경로를 확인해주세요.")
        return
    
    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)

    # 2. 테스트 이미지 목록 수집 (학습된 클래스만)
    test_root = Path('dataset/test')
    if not test_root.exists():
        print(f"에러: 테스트 폴더를 찾을 수 없습니다: {test_root}")
        return

    # 모델이 배운 클래스 목록 (소문자로 변환하여 비교 준비)
    trained_classes = [name.lower() for name in model.names.values()]
    print(f"모델이 학습한 클래스: {trained_classes}")

    all_test_images = []
    
    # test/ 하위의 각 폴더(클래스)를 순회
    for class_folder in test_root.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            
            # 모델이 학습하지 않은 클래스 폴더는 건너뜀
            if class_name.lower() not in trained_classes:
                continue

            # 지원하는 이미지 확장자 검색
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
            images = []
            for ext in extensions:
                images.extend(list(class_folder.glob(ext)))
            
            for img_path in images:
                all_test_images.append({
                    'path': str(img_path),
                    'label': class_name
                })

    if not all_test_images:
        print(f"에러: 모델이 학습한 클래스({trained_classes})에 해당하는 이미지를 'dataset/test'에서 찾을 수 없습니다.")
        return

    # 3. 랜덤 샘플링 (100개 또는 전체 이미지 수가 적으면 전체)
    sample_size = min(num_samples, len(all_test_images))
    sampled_data = random.sample(all_test_images, sample_size)
    print(f"총 {len(all_test_images)}개 중 {sample_size}개 이미지를 랜덤으로 추출하여 테스트를 시작합니다...\n")

    # 4. 테스트 수행
    correct_count = 0
    results_summary = []

    for item in sampled_data:
        img_path = item['path']
        true_label = item['label'].strip()

        # 예측 실행
        results = model.predict(source=img_path, verbose=False)
        result = results[0]
        
        # 모델이 학습한 클래스 이름 리스트 가져오기
        class_names = result.names  # 예: {0: 'calculator', 1: 'mouse', ...}
        
        # 가장 높은 확률의 클래스 인덱스와 이름
        top1_idx = result.probs.top1
        pred_label = class_names[top1_idx].strip()
        conf = result.probs.top1conf.item()

        # 대소문자 구분 없이 비교 (정확도 향상을 위해)
        is_correct = (true_label.lower() == pred_label.lower())
        if is_correct:
            correct_count += 1
        
        results_summary.append({
            'path': Path(img_path).name,
            'true': true_label,
            'pred': pred_label,
            'conf': conf,
            'ok': is_correct
        })

    # 5. 최종 결과 출력
    print("-" * 85)
    print(f"{'파일명':<40} | {'정답':<12} | {'예측':<12} | {'결과'}")
    print("-" * 85)
    
    for r in results_summary:
        status = "✅" if r['ok'] else "❌"
        # 파일명이 너무 길면 잘라서 출력
        display_name = (r['path'][:37] + '..') if len(r['path']) > 37 else r['path']
        print(f"{display_name:<40} | {r['true']:<12} | {r['pred']:<12} | {status} ({r['conf']:.2%})")

    accuracy = (correct_count / sample_size) * 100
    print("-" * 85)
    print(f"테스트 결과 요약:")
    print(f" - 전체 테스트 개수: {sample_size}")
    print(f" - 맞은 개수: {correct_count}")
    print(f" - 틀린 개수: {sample_size - correct_count}")
    print(f" - 최종 정확도 (Accuracy): {accuracy:.2f}%")
    print("-" * 85)

if __name__ == "__main__":
    # 인자로 샘플링 개수를 조절할 수 있습니다 (기본 100개)
    run_random_test(100)
