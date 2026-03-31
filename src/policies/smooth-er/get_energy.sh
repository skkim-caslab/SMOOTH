#!/bin/bash

# energy_out 디렉토리 내의 block...으로 시작하는 모든 디렉토리를 대상으로 합니다.
# 만약 특정 패턴만 원하시면 "energy_out/block*" 로 지정하세요.
DIRS=$(ls -d energy_out/block*)

for DIR in $DIRS; do
    # 디렉토리인지 다시 한 번 확인
    if [ -d "$DIR" ]; then
        echo "----------------------------------------------------------"
        echo "Processing directory: $DIR"
        
        # 해당 디렉토리 내의 .out 파일들 찾기
        # _overhead.out으로 끝나는 파일은 제외하고 순수 결과 파일만 루프
        for FILE in "$DIR"/*.out; do
            # 파일이 존재하지 않는 경우(빈 디렉토리) 예외 처리
            [ -e "$FILE" ] || continue
            
            # 이미 처리된 _overhead.out 파일은 건너뜀
            if [[ "$FILE" == *"_overhead.out" ]]; then
                continue
            fi

            # 파일명에서 .out을 제거하고 _overhead.out을 붙여 저장
            TARGET="${FILE%.out}_overhead.out"

            # SKKIM OVERHEAD 라인 추출
            grep "SKKIM OVERHEAD" "$FILE" > "$TARGET"
            
            # 결과 확인을 위한 간단한 출력
            echo "  > Extracted: $(basename "$FILE") -> $(basename "$TARGET")"
        done
    fi
done

echo "----------------------------------------------------------"
echo "All extractions are complete for all block sizes."
