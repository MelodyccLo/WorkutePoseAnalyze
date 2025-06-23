import pandas as pd
import json
import ast # 用於安全地評估字串，將其轉換為 Python 資料結構
import os  # 用於處理檔案路徑

# 檔案路徑 - 請將此替換為您的實際 .xlsx 檔案路徑
excel_file = "/Users/melodylo/Desktop/Computer/WorkutePoseAnalyze/test.xlsx"

def convert_excel_sheets_to_json(excel_file_path: str) -> dict:
    """
    從單一 Excel (.xlsx) 檔案的不同工作表讀取資料，並將其轉換為指定的 JSON 格式。

    Args:
        excel_file_path (str): 包含所有運動資料的工作簿 (.xlsx) 檔案路徑。
                                預期包含以下三個工作表：
                                - 'Exercise General Info'
                                - 'Exercise Configurations'
                                - 'Angle Constraints'

    Returns:
        dict: 轉換後的 JSON 資料作為 Python 字典。
    """

    # 定義工作表名稱
    sheet_name_general_info = 'Exercise General Info'
    sheet_name_configurations = 'Exercise Configurations'
    sheet_name_angle_constraints = 'Angle Constraints'

    # 1. 讀取 Excel 檔案中的各個工作表
    try:
        # 使用 sheet_name 參數讀取特定工作表
        df_general_info = pd.read_excel(excel_file_path, sheet_name=sheet_name_general_info)
        df_configurations = pd.read_excel(excel_file_path, sheet_name=sheet_name_configurations)
        df_angle_constraints = pd.read_excel(excel_file_path, sheet_name=sheet_name_angle_constraints)

    except FileNotFoundError:
        print(f"錯誤：找不到 Excel 檔案 - {excel_file_path}")
        return {}
    except ValueError as e:
        print(f"錯誤：找不到指定的工作表。請確認 Excel 檔案 '{excel_file_path}' 中存在 '{sheet_name_general_info}', '{sheet_name_configurations}' 和 '{sheet_name_angle_constraints}' 工作表。詳情：{e}")
        return {}
    except Exception as e:
        print(f"讀取 Excel 檔案時發生錯誤：{e}")
        return {}

    # 初始化最終的 JSON 結構
    final_json = {}

    # 2. 處理 Exercise General Info 工作表
    # 假設 Exercise General Info 工作表只有一行資料
    if not df_general_info.empty:
        general_info = df_general_info.iloc[0]
        # 修正欄位名稱以匹配 Excel 中的實際名稱 (首字母大寫，包含空格)
        final_json["hint"] = general_info.get("Hint", "")
        final_json["title"] = general_info.get("Title", "")
        final_json["version"] = str(general_info.get("Version", "1.0.0")) # 確保為字串
        final_json["bgm_type"] = general_info.get("BGM Type", "EXERCISE_WORKOUT")
        # 將 'Required' 欄位從字串（如 'TRUE', 'FALSE'）轉換為布林值
        final_json["required"] = str(general_info.get("Required", "FALSE")).upper() == "TRUE"
        final_json["description"] = general_info.get("Description", "")
        # 根據您提供的 JSON 範例，introduction.context 的內容與 description 相同
        final_json["introduction"] = {"context": general_info.get("Description", "")}
        final_json["banner_color_code"] = general_info.get("Banner Color Code", "\"#EA0000FF\"")
        # 額外儲存 Key 欄位用於檔案命名
        final_json["file_key"] = general_info.get("Key", "default_exercise_name")
    else:
        print("警告：'Exercise General Info' 工作表為空或無有效資料。將使用預設值。")
        # 提供預設值以避免錯誤
        final_json["hint"] = ""
        final_json["title"] = ""
        final_json["version"] = "1.0.0"
        final_json["bgm_type"] = "EXERCISE_WORKOUT"
        final_json["required"] = False
        final_json["description"] = ""
        final_json["introduction"] = {"context": ""}
        final_json["banner_color_code"] = "\"#EA0000FF\""
        final_json["file_key"] = "default_exercise_name" # 預設檔案命名鍵


    # 3. 處理 Angle Constraints 工作表
    # 建立一個巢狀字典，用於根據 Exercise Name 和 Status Name 儲存 JustUnits
    angle_constraints_data = {}
    for index, row in df_angle_constraints.iterrows():
        # 修正欄位名稱以匹配 Excel 中的實際名稱
        exercise_name = row.get("Exercise Name")
        status_name = row.get("Status Name")
        body_part_index = row.get("Body Part Index")
        angle_min = row.get("Angle Min")
        angle_max = row.get("Angle Max")

        # 檢查關鍵欄位是否為 NaN，如果是則跳過該行
        if pd.isna(exercise_name) or pd.isna(status_name) or pd.isna(body_part_index) or pd.isna(angle_min) or pd.isna(angle_max):
            continue

        # 確保 exercise_name 和 status_name 是字串，以避免字典鍵錯誤
        exercise_name = str(exercise_name)
        status_name = str(status_name)

        if exercise_name not in angle_constraints_data:
            angle_constraints_data[exercise_name] = {}
        if status_name not in angle_constraints_data[exercise_name]:
            angle_constraints_data[exercise_name][status_name] = []

        try:
            angle_constraints_data[exercise_name][status_name].append({
                "angleMax": int(angle_max),
                "angleMin": int(angle_min),
                "BodyPartIndex": int(body_part_index)
            })
        except ValueError as e:
            print(f"除錯：在 'Angle Constraints' 工作表第 {index} 行轉換角度或索引時出錯：{e}。行資料：{row.to_dict()}")
            continue


    # 4. 處理 Exercise Configurations 工作表
    angle_configs = []
    for index, row in df_configurations.iterrows():
        # 修正欄位名稱以匹配 Excel 中的實際名稱
        exercise_name = row.get("Exercise Name")
        combine_letter = row.get("Combine Letter", "@") # 提供預設值

        # 將逗號分隔的字串轉換為列表，並嘗試將元素轉換為整數或字串
        def parse_list_from_string(s, item_type=str):
            if pd.isna(s) or s == "":
                return []
            try:
                # 嘗試使用 ast.literal_eval 處理更複雜的列表字串（例如 "[1, 2, 3]" 或 "['a', 'b']"）
                # 如果是簡單的逗號分隔字串，則回退到 split
                if isinstance(s, str) and s.strip().startswith('[') and s.strip().endswith(']'):
                    return ast.literal_eval(s)
                elif isinstance(s, str):
                    return [item_type(x.strip()) for x in s.split(',') if x.strip()]
                else: # 如果不是字串類型，例如數字或 NaN
                    return []
            except (ValueError, SyntaxError) as e:
                # print(f"除錯：解析列表字串時出錯（類型 {item_type.__name__}）：{s}，錯誤：{e}")
                if isinstance(s, str):
                    # 回退到簡單的逗號分隔
                    return [item_type(x.strip()) for x in s.split(',') if x.strip()]
                return []


        focus_pos_only_list = parse_list_from_string(row.get("Focus Pos Only List"), int)
        focus_lines_index_list = parse_list_from_string(row.get("Focus Lines Index List"), int)
        count_match_string_list = parse_list_from_string(row.get("Count Match String List"), str)
        expect_status_change_process = parse_list_from_string(row.get("Expect Status Change Process"), str)

        current_status_list = []
        # 確保 exercise_name 是字串，以匹配 angle_constraints_data 的鍵
        if str(exercise_name) in angle_constraints_data:
            for status_name, just_units in angle_constraints_data[str(exercise_name)].items():
                current_status_list.append({
                    "JustUnits": just_units,
                    "statusName": status_name
                })
        else:
            # print(f"除錯：在 'Exercise Configurations' 工作表第 {index} 行，找不到 '{exercise_name}' 的角度限制資料。")
            pass # 這裡可以選擇印出警告，但為了簡潔先關閉

        angle_configs.append({
            "statusList": current_status_list,
            "exerciseName": exercise_name,
            "combineLetter": combine_letter,
            "focusPosOnlyList": focus_pos_only_list,
            "focusLinesIndexList": focus_lines_index_list,
            "countMatchStringList": count_match_string_list,
            "expectStatusChangeProcess": expect_status_change_process
        })

    final_json["angle_configs"] = angle_configs

    return final_json


# 執行轉換
output_json_data = convert_excel_sheets_to_json(excel_file)

# 動態定義輸出檔案的名稱和路徑
output_file_name = "exercise_data.txt" # 預設檔案名
if output_json_data and "file_key" in output_json_data:
    # 清理檔案名稱，替換掉任何不安全的字元
    safe_file_key = "".join(c for c in output_json_data["file_key"] if c.isalnum() or c in ('-', '_')).strip()
    if safe_file_key:
        output_file_name = f"{safe_file_key}.txt"
    # 從最終 JSON 中移除 file_key 欄位，因為它只是用於檔案命名
    del output_json_data["file_key"]

# 獲取當前工作目錄，將輸出檔案儲存在同一目錄
output_file_path = os.path.join(os.getcwd(), output_file_name)


# 輸出 JSON 到控制台（或檔案）
if output_json_data:
    final_json_string = json.dumps(output_json_data, indent=4, ensure_ascii=False)

    # 將 JSON 字串寫入 .txt 檔案
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(final_json_string)
        print(f"\n成功將 JSON 資料儲存到檔案：{output_file_path}")
    except Exception as e:
        print(f"\n儲存 JSON 檔案時發生錯誤：{e}")

    print(final_json_string) # 仍然在控制台打印一份，方便檢查
    
else:
    print("\n--- JSON 生成失敗或結果為空，未儲存檔案 ---")
