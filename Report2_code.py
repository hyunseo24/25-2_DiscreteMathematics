import copy

def get_matrix(n):
    #(추가 기능 적용)사용자로부터 n x n 관계 행렬을 입력
    print(f"집합 A = {{1, 2, 3, 4, 5}}에 대한 5x5 관계 행렬을 입력합니다.")
    print("각 행의 원소를 띄어쓰기로 구분하여 5개씩 입력하세요. (예: 1 0 0 1 0)")
    matrix = []
    for i in range(n):
        while True:
            try:
                row_input = input(f"  {i+1}번째 행 입력: ")
                row_str_list = row_input.split()

                # 1. 개수 검사
                if len(row_str_list) != n:
                    print(f"  [오류] 정확히 {n}개의 숫자를 입력해야 합니다.")
                    continue
                
                row = [int(x) for x in row_str_list]

                # 2. 0 또는 1 인지 검사 (★추가된 기능★)
                is_valid = True
                for val in row:
                    if val not in (0, 1):
                        is_valid = False
                        break
                
                if not is_valid:
                    print("  [오류] 관계 행렬의 원소는 0 또는 1이어야 합니다.")
                    continue

                # 모든 검사 통과
                matrix.append(row)
                break

            except ValueError:
                print("  [오류] 숫자만 입력하세요.")
    return matrix

def print_matrix(matrix, title=""):

    if title:
        print(f"\n{title}")
    print("-" * (len(title) if title else 5))
    for row in matrix:
        print("  " + " ".join(map(str, row)))
    print("-" * (len(title) if title else 5))

# --- 2. 동치 관계 판별 기능 ---

def is_reflexive(matrix, n):
    for i in range(n):
        if matrix[i][i] == 0:
            return False
    return True

def is_symmetric(matrix, n):

    for i in range(n):
        for j in range(i + 1, n): # 상단 삼각행렬만 검사
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

def is_transitive(matrix, n):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # (i,j) 관계가 있고 (j,k) 관계가 있는데
                if matrix[i][j] == 1 and matrix[j][k] == 1:
                    # 추이 관계가 아님
                    if matrix[i][k] == 0:
                        return False
    return True

# --- 3. 동치류 출력 기능 ---

def print_equivalence_classes(matrix, n, elements):
    print("\n[동치류 출력]")
    processed = [False] * n  # 동치류가 출력된 원소 확인

    for i in range(n):
        if processed[i]:
            continue

        eq_class = []
        for j in range(n):
            # 관계가 있음
            if matrix[i][j] == 1:
                eq_class.append(elements[j])
                processed[j] = True # j는 i와 같은 동치류이므로 나중에 또 검사할 필요 없음

        print(f"  {elements[i]}의 동치류: {sorted(eq_class)}")

# --- 4. 폐포 구현 기능 ---

def reflexive_closure(matrix, n):
    # 원본 수정을 방지하기 위해 깊은 복사 사용
    r_matrix = copy.deepcopy(matrix)
    for i in range(n):
        r_matrix[i][i] = 1
    return r_matrix

def symmetric_closure(matrix, n):
    s_matrix = copy.deepcopy(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if s_matrix[i][j] == 1:
                s_matrix[j][i] = 1
            elif s_matrix[j][i] == 1:
                s_matrix[i][j] = 1
    return s_matrix

def transitive_closure(matrix, n):
    t_matrix = copy.deepcopy(matrix)
    # R* = U R^n
    for k in range(n): # 중간 정점 k
        for i in range(n): # 시작 정점 i
            for j in range(n):
                # i->k와 k->j 경로가 있다면 i->j 경로 추가
                t_matrix[i][j] = t_matrix[i][j] or (t_matrix[i][k] and t_matrix[k][j])
    return t_matrix

# --- 헬퍼 함수 및 메인 로직 ---

def check_all_properties(matrix, title, n, elements): # 모든 속성을 한 번에 검사하고 결과를 출력하는 헬퍼 함수

    print(f"\n{title}")
    
    r = is_reflexive(matrix, n)
    s = is_symmetric(matrix, n)
    t = is_transitive(matrix, n)

    print(f"  1. 반사 관계 (Reflexive):  {'O' if r else 'X'}")
    print(f"  2. 대칭 관계 (Symmetric):  {'O' if s else 'X'}")
    print(f"  3. 추이 관계 (Transitive): {'O' if t else 'X'}")

    if r and s and t:
        print("\n  >> 결과: 이 관계는 동치 관계입니다.")
        print_equivalence_classes(matrix, n, elements)
        return True
    else:
        print("\n  >> 결과: 이 관계는 동치 관계가 아닙니다.")
        return False

def process_closures(matrix, n, elements):
    #(4번 조건) 동치 관계가 아닐 경우, 필요한 폐포를 순차적으로 적용. 동치 폐포(Equivalence Closure)는 T(S(R(R))
    print("\n" + "="*40)
    print("  동치 관계가 아니므로, 동치 폐포를 생성합니다.")
    print("  (반사 -> 대칭 -> 추이 폐포 순으로 적용)")
    print("="*40)

    # 초기 상태 확인
    r = is_reflexive(matrix, n)
    s = is_symmetric(matrix, n)
    t = is_transitive(matrix, n)

    final_matrix = copy.deepcopy(matrix)

    # 1. 반사 폐포 적용
    if not r:
        print("\n--- 1. 반사 폐포 (Reflexive Closure) 적용 ---")
        temp_r = reflexive_closure(final_matrix, n)
        print_matrix(final_matrix, "  변환 전 (Before)")
        print_matrix(temp_r, "  변환 후 (After)")
        final_matrix = temp_r # 다음 단계로 전달
    else:
        print("\n--- 1. 반사 관계 (이미 만족함) ---")

    # 2. 대칭 폐포 적용
    s_current = is_symmetric(final_matrix, n)
    if not s_current:
        print("\n--- 2. 대칭 폐포 (Symmetric Closure) 적용 ---")
        temp_s = symmetric_closure(final_matrix, n)
        print_matrix(final_matrix, "  변환 전 (Before)")
        print_matrix(temp_s, "  변환 후 (After)")
        final_matrix = temp_s
    else:
         print("\n--- 2. 대칭 관계 (이미 만족함) ---")

    # 3. 추이 폐포 적용
    t_current = is_transitive(final_matrix, n)
    if not t_current:
        print("\n--- 3. 추이 폐포 (Transitive Closure) 적용 ---")
        temp_t = transitive_closure(final_matrix, n)
        print_matrix(final_matrix, "  변환 전 (Before)")
        print_matrix(temp_t, "  변환 후 (After)")
        final_matrix = temp_t
    else:
        print("\n--- 3. 추이 관계 (이미 만족함) ---")

    # 4. 모든 폐포 적용 후 최종 판별
    print("\n" + "="*40)
    print("  모든 폐포 적용, 최종 행렬을 판별.")
    print("="*40)
    check_all_properties(final_matrix, "최종 동치 폐포(Equivalence Closure) 판별", n, elements)


def main():
    N = 5
    A = [1, 2, 3, 4, 5] 

    # 관계 행렬 입력
    matrix = get_matrix(N)
    print_matrix(matrix, "입력된 관계 행렬 (R)")

    # 동치 관계 판별 및 동치류 출력
    is_equivalence = check_all_properties(matrix, "--- 1. 최초 관계 판별 ---", N, A)

    # 동치 관계가 아닐 경우 폐포 구현 및 재판별
    if not is_equivalence:
        process_closures(matrix, N, A)

if __name__ == "__main__":
    main()
