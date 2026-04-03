import os
import re
from deep_translator import GoogleTranslator

def interactive_korean_translator(directory):
    korean_pattern = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
    translator = GoogleTranslator(source='ko', target='en')

    for root, _, files in os.walk(directory):
        for file in files:
            if file == "check_all.py":
                continue
                
            if not file.endswith(('.py', '.c', '.cpp', '.h', '.sh')): 
                continue
                
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                file_modified = False
                new_lines = []
                
                for i, line in enumerate(lines):
                    if korean_pattern.search(line):
                        original_line = line.rstrip('\n')
                        
                        # ⭐ [핵심 수정] 앞쪽 들여쓰기(공백/탭) 추출 및 분리
                        match = re.match(r'^(\s*)(.*)', original_line)
                        indent = match.group(1) # 들여쓰기 보관
                        text_to_translate = match.group(2) # 실제 번역할 텍스트
                        
                        try:
                            # 텍스트만 번역 후, 보관했던 들여쓰기를 다시 합침
                            translated_text = translator.translate(text_to_translate)
                            translated_line = indent + translated_text
                            
                            print(f"\n[{filepath} - Line {i+1}]")
                            print(f" 🔴 원본: {original_line}")
                            print(f" 🟢 번역: {translated_line}")
                            
                            user_input = input(" ➡️ 영어로 변경하시겠습니까? (y:변경 / n:유지 / q:종료): ").strip().lower()
                            
                            if user_input == 'q':
                                if file_modified:
                                    remaining_lines = lines[i+1:]
                                    with open(filepath, 'w', encoding='utf-8') as f:
                                        f.writelines(new_lines + [line] + remaining_lines)
                                    print(f"💾 중단 전까지의 변경사항을 [{file}]에 저장했습니다.")
                                print("🛑 스크립트를 완전히 종료합니다.")
                                return
                                
                            elif user_input == 'y':
                                new_lines.append(translated_line + '\n')
                                file_modified = True
                            else:
                                new_lines.append(line)
                                
                        except Exception as trans_e:
                            print(f" ⚠️ [번역 에러] {trans_e}")
                            new_lines.append(line)
                    else:
                        new_lines.append(line)

                if file_modified:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                    print(f"\n✅ [{filepath}] 저장 완료.")
                    print("-" * 50)
                    
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

# 스크립트 실행
interactive_korean_translator("./")
