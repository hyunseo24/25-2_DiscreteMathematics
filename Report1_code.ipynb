 '''
 1. 행렬 입력 기능

  -- 사용자로부터 정수 n을 입력받아 n × n 크기의 정방행렬을 행 단위로 입력받을 수 있어야 함

  -- 입력 데이터는 리스트(2차원 배열)로 저장

 2. 행렬식을 이용한 역행렬 계산 기능

  -- 행렬식을 사용하여 주어진 행렬의 역행렬을 계산하는 함수를 구현

  -- 역행렬이 존재하지 않을 경우(행렬식이 0인 경우) 오류 메시지를 출력

 3. 가우스-조던 소거법(Gauss-Jordan elimination)을 이용한 역행렬 계산 기능

  -- 가우스-조던 소거법을 사용하여 동일한 행렬의 역행렬을 계산하는 함수를 구현

  -- 행렬식과 마찬가지로 역행렬이 존재하지 않는 경우 예외 처리 포함

 4. 결과 출력 및 비교 기능

  -- 두 방법(행렬식, 가우스-조던)으로 계산한 역행렬을 각각 출력

  -- 두 결과가 동일한지 비교하여 결과 메시지를 출력
'''

import time
import copy



# 구성 조건 4: 결과 출력 기능
def print_matrix(matrix, name="Matrix"):
    if matrix is None:
        print(f"{name} is None.")
        return
    print(f"--- {name} ---")
    for row in matrix:
        # 소수점 4자리까지 출력
        print("  ".join(f"{elem:10.4f}" for elem in row))
    print("-" * (len(matrix[0]) * 12))



# 구성 조건 2: 행렬식을 이용한 역행렬 계산 기능
def get_minor(matrix, i, j):
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

def calculate_determinant(matrix):
    # 2x2 행렬, ad - bc
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determinant = 0
    # 첫 번째 행을 기준으로 여인수 전개
    for c in range(len(matrix)):
        determinant += ((-1)**c) * matrix[0][c] * calculate_determinant(get_minor(matrix, 0, c))
    return determinant

def inverse_by_determinant(matrix):
    n = len(matrix)
    # 1x1 행렬
    if n == 1:
        determinant = matrix[0][0]
        if determinant == 0:
            return None
        return [[1/determinant]]
        
    determinant = calculate_determinant(matrix)

    # 행렬식이 0이면 역행렬x
    if determinant == 0:
        return None

    # 여인수 행렬 생성
    cofactors = []
    for r in range(n):
        cofactor_row = []
        for c in range(n):
            minor = get_minor(matrix, r, c)
            cofactor_row.append(((-1)**(r+c)) * calculate_determinant(minor))
        cofactors.append(cofactor_row)

    # 수반 행렬 생성
    adjugate = list(map(list, zip(*cofactors)))
    # 역행렬 계산: (1/행렬식) * 수반 행렬
    inverse_matrix = [[elem / determinant for elem in row] for row in adjugate]

    return inverse_matrix



# 구성 조건 3: 가우스-조던 소거법을 이용한 역행렬 계산 기능
def inverse_by_gauss_jordan(matrix):

    n = len(matrix)
    # 원본 행렬을 수정x
    mat = copy.deepcopy(matrix)

    # 첨가 행렬 생성
    identity = [[float(i == j) for i in range(n)] for j in range(n)]
    augmented_matrix = [mat[i] + identity[i] for i in range(n)]

    # Forward Elimination 수행
    for i in range(n):
        if augmented_matrix[i][i] == 0.0:
            swap_row = -1
            for j in range(i + 1, n):
                if augmented_matrix[j][i] != 0.0:
                    swap_row = j
                    break
            
            if swap_row == -1:
                return None

            augmented_matrix[i], augmented_matrix[swap_row] = augmented_matrix[swap_row], augmented_matrix[i]
        
        # 피벗 원소를 1로
        pivot = augmented_matrix[i][i]
        for j in range(i, n * 2):
            augmented_matrix[i][j] /= pivot

        # 현재 열의 다른 모든 원소를 0
        for j in range(n):
            if i != j:
                ratio = augmented_matrix[j][i]
                for k in range(i, n * 2):
                    augmented_matrix[j][k] -= ratio * augmented_matrix[i][k]

    # 오른쪽 부분인 역행렬만 추출 return
    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix



# 구성 조건 4: 결과 출력 및 비교 기능 (+ 추가 기능)
def are_matrices_equal(matrix1, matrix2, tolerance=1e-9): # 허용오차 기능

    if matrix1 is None or matrix2 is None:
        return False
    for r in range(len(matrix1)):
        for c in range(len(matrix1[0])):
            if abs(matrix1[r][c] - matrix2[r][c]) > tolerance:
                return False
    return True



# main 함수
def main():
    print("=" * 50)
    print("역행렬 계산 프로그램 (행렬식 vs 가우스-조던 소거법)")
    print("=" * 50)

    while True:
        try:
            n = int(input("행렬의 크기 n을 입력하세요: "))
            if n > 0:
                break
            else:
                print("n은 0보다 큰 정수여야 합니다.")
        except ValueError:
            print("올바른 정수를 입력해주세요.")

    print(f"\n{n} x {n} 행렬의 원소를 행 단위로 입력하세요.")
    print("각 원소는 공백으로 구분합니다. (예: 1 2 3)")
    
    matrix = []
    for i in range(n):
        while True:
            row_input = input(f"{i+1}번째 행 입력: ")
            try:
                row = [float(x) for x in row_input.split()]
                if len(row) == n:
                    matrix.append(row)
                    break
                else:
                    print(f"정확히 {n}개의 원소를 입력해야 합니다.")
            except ValueError:
                print("숫자 형식의 데이터만 입력해주세요.")

    print("\n입력된 행렬:")
    print_matrix(matrix, "Original Matrix (A)")
    
    # 추가 기능 1: 시간 측정 기능
    # 행렬식을 이용한 역행렬 계산 및 시간 측정
    print("\n[방법 1: 행렬식을 이용한 계산]")
    start_time = time.time()
    inv_det = inverse_by_determinant(matrix)
    end_time = time.time()
    time_det = end_time - start_time

    if inv_det is None:
        print("행렬식이 0이므로 역행렬이 존재하지 않습니다.")
    else:
        print_matrix(inv_det, "Inverse by Determinant")
        print(f"계산 시간: {time_det:.6f} 초")

    # 가우스-조던 소거법을 이용한 역행렬 계산 및 시간 측정
    print("\n[방법 2: 가우스-조던 소거법을 이용한 계산]")
    start_time = time.time()
    inv_gj = inverse_by_gauss_jordan(matrix)
    end_time = time.time()
    time_gj = end_time - start_time

    if inv_gj is None:
        print("행렬이 특이 행렬(Singular Matrix)이므로 역행렬이 존재하지 않습니다.")
    else:
        print_matrix(inv_gj, "Inverse by Gauss-Jordan")
        print(f"계산 시간: {time_gj:.6f} 초")
    
    # 추가 기능 2: 결과 비교 및 분석
    print("\n[결과 비교]")
    if inv_det is not None and inv_gj is not None:
        if are_matrices_equal(inv_det, inv_gj):
            print(">> 두 방법으로 계산한 역행렬은 동일합니다.")
        else:
            print(">> 두 방법으로 계산한 역행렬이 다릅니다. (계산 오류 확인 필요)")
        
        print("\n[성능 비교]")
        if time_det < time_gj:
            print(f">> '행렬식' 방법이 {time_gj - time_det:.6f} 초 더 빨랐습니다.")
        else:
            print(f">> '가우스-조던 소거법'이 {time_det - time_gj:.6f} 초 더 빨랐습니다.")
        print("(참고: 일반적으로 행렬 크기가 커질수록 가우스-조던 소거법이 더 효율적입니다.)")

    elif inv_det is None and inv_gj is None:
        print(">> 두 방법 모두 역행렬이 존재하지 않는다고 올바르게 판단했습니다.")
    else:
        print(">> 한 방법에서는 역행렬을 찾았으나 다른 방법에서는 찾지 못했습니다. (알고리즘 구현 확인 필요)")
        
    print("\n" + "=" * 50)
    print("프로그램을 종료합니다.")
    print("=" * 50)


if __name__ == '__main__':
    main()
