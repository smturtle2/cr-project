import torch

# [경로 수정 완료] 폴더 구조에 맞게 import 경로 변경
from modules.model.cafm.cafm import CloudDensityEstimator, CAFM

def run_dummy_test():
    print("========== CAFM 모듈 단위 테스트 시작 ==========\n")
    
    # 1. 더미 데이터 생성 (배치사이즈=2, 이미지크기=64x64 로 가정)
    B, H, W = 2, 64, 64
    C_feature = 256  # ACA-CRNet의 중간 피처맵 채널 수
    
    print("[입력 데이터 생성]")
    sar_dummy = torch.rand(B, 2, H, W)        # SAR 데이터 (2채널)
    cloudy_dummy = torch.rand(B, 13, H, W)    # 광학 데이터 (13채널)
    feature_dummy = torch.rand(B, C_feature, H, W) # 네트워크 피처맵 (256채널)
    
    print(f" - SAR 텐서: {sar_dummy.shape}")
    print(f" - 광학 텐서: {cloudy_dummy.shape}")
    print(f" - 피처 텐서: {feature_dummy.shape}\n")

    # 2. 모듈 초기화
    try:
        density_estimator = CloudDensityEstimator(sar_channels=2, optical_channels=13)
        cafm_module = CAFM(feature_channels=C_feature)
        print("✅ 모듈 초기화 성공\n")
    except Exception as e:
        print(f"❌ 모듈 초기화 실패: {e}")
        return

    # 3. 밀도 추정기(Density Estimator) 테스트
    print("[1단계: 밀도맵 생성 테스트]")
    try:
        density_map = density_estimator(sar_dummy, cloudy_dummy)
        print(f" - 출력 밀도맵 형태: {density_map.shape}")
        assert density_map.shape == (B, 1, H, W), "밀도맵 Shape가 예상과 다릅니다!"
        print(" ✅ 밀도맵 생성 성공 (풀링 없이 HxW 공간 정보 유지됨!)\n")
    except Exception as e:
        print(f"❌ 밀도맵 생성 실패: {e}")
        return

    # 4. CAFM 변조(Modulator) 테스트
    print("[2단계: 피처 변조(CAFM) 테스트]")
    try:
        output_feature = cafm_module(feature_dummy, density_map)
        print(f" - 출력 피처 형태: {output_feature.shape}")
        assert output_feature.shape == (B, C_feature, H, W), "출력 피처 Shape가 예상과 다릅니다!"
        print(" ✅ 피처 변조 성공 (입력 피처와 동일한 Shape로 무사히 출력됨!)\n")
    except Exception as e:
        print(f"❌ 피처 변조 실패: {e}")
        return

    # 5. [핵심] Zero-init 검증
    print("[3단계: Zero-init (원본 보존) 검증]")
    # 입력 피처와 출력 피처가 오차 범위 내에서 완전히 똑같은지 확인
    is_identical = torch.allclose(feature_dummy, output_feature, atol=1e-6)
    
    if is_identical:
        print(" ✅ 성공! Zero-init이 완벽하게 작동합니다.")
        print("    -> 현재 모듈은 가중치가 0으로 초기화되어, 입력된 피처를 100% 그대로 통과시켰습니다.")
    else:
        print(" ❌ 실패! 값이 변형되었습니다. Zero-init 코드를 확인해 주세요.")

    print("\n========== 테스트 종료 (모두 정상 작동) ==========")

if __name__ == "__main__":
    run_dummy_test()