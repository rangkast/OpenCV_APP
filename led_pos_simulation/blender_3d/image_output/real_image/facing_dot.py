import math  # for math.cos and math.radians

def calculate_facing_dot(RIFT_LED_ANGLE):
    # 각도를 라디안으로 변환합니다.
    angle_radians = math.radians(RIFT_LED_ANGLE)
    
    # 각도의 코사인 값을 계산합니다.
    facing_dot = math.cos(angle_radians)
    
    return facing_dot

# 테스트
angles = [50,40]
for angle in angles:
    facing_dot = calculate_facing_dot(angle)
    print(f"Facing dot for RIFT_LED_ANGLE {angle}: {facing_dot:.4f}")   
