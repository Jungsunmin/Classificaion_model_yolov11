import sys
from ultralytics import YOLO
import os

def predict_image(image_path):
    # 1. 학습된 최적 모델 로드
    model_path = 'runs/classify/flea_market_cls/test_run_5_classes-2/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"에러: 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요: {model_path}")
        return

    model = YOLO(model_path)

    # 2. 이미지 파일 존재 여부 확인
    if not os.path.exists(image_path):
        print(f"에러: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    # 3. 예측 실행
    print(f"\n입력 이미지: {image_path}")
    results = model.predict(source=image_path, verbose=False)

    # 4. 결과 출력
    for result in results:
        # 가장 높은 확률을 가진 클래스의 인덱스와 이름 가져오기
        top1_idx = result.probs.top1
        top1_conf = result.probs.top1conf.item()
        top1_label = result.names[top1_idx]

        print("-" * 30)
        print(f"최종 예측 결과: {top1_label}")
        print(f"신뢰도(Confidence): {top1_conf:.2%}")
        print("-" * 30)
        
        # 전체 클래스별 확률 출력 (상위 5개)
        print("전체 클래스별 확률:")
        for idx in result.probs.top5:
            name = result.names[idx]
            conf = result.probs.data[idx].item()
            print(f" - {name}: {conf:.2%}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 predict.py <이미지_경로>")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)
