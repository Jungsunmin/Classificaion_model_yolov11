from ultralytics import YOLO
import os

def train_model():
    # 1. 모델 설정 (5개 클래스 테스트용으로 가장 가벼운 nano 모델 사용)
    model = YOLO('yolo11s-cls.pt')

    # 2. 데이터셋 경로 설정
    # 현재 디렉토리의 dataset 폴더를 참조합니다.
    dataset_path = os.path.abspath('dataset')

    # 3. 모델 학습
    results = model.train(
        data=dataset_path,
        epochs=50,          # 테스트용이므로 50회 설정 (필요시 수정)
        imgsz=224,          # 기본 이미지 크기
        batch=16,           # 메모리 사양에 따라 조절 (8, 16, 32 등)
        device='mps',       # Apple Silicon(M1/M2/M3) GPU 사용
        project='flea_market_cls',
        name='test_run_5_classes'
    )

    print("학습 완료!")
    return results

if __name__ == "__main__":
    train_model()
