import os
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---- CONFIG ----
IMAGE_FOLDER = "C:\\Users\\sahil\\Desktop\\Meal_final\\crops"
OUTPUT_DATA = {}

def setup_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
        Object.defineProperty(navigator, 'webdriver', {
          get: () => undefined
        });
        """
    })
    return driver

def upload_and_extract(driver, image_path):
    driver.get("https://lens.google.com/upload")

    try:
        # Upload image
        file_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]'))
        )
        file_input.send_keys(image_path)
        print(f"[+] Uploaded: {os.path.basename(image_path)}")

        # Wait and prompt
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Add to your search']"))
        ).send_keys("list the item in the image" + Keys.ENTER)

        time.sleep(5)

        # Collect all spans in the 4th block (fallback if AI Overview not labeled)
        spans = WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((
                By.XPATH,
                "//div[@class='LT6XE']//span"
            ))
        )

        return [span.text.strip() for span in spans if span.text.strip()]
    except Exception as e:
        print(f"[-] Failed to process {os.path.basename(image_path)}: {e}")
        return ["Error or no output"]

def batch_process_images(folder_path):
    driver = setup_driver()
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        results = upload_and_extract(driver, img_path)
        seen = set()
        unique_results = []
        for item in results:
            if item not in seen:
                seen.add(item)
                unique_results.append(item)
        OUTPUT_DATA[img_name] = unique_results
        #OUTPUT_DATA[img_name] = results
        time.sleep(2)  # polite delay to avoid rate-limiting

    driver.quit()

    # Save results to JSON
    output_json_path = os.path.join(folder_path, "lens_output.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(OUTPUT_DATA, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_json_path}")

    # print("\n\n=== FINAL OUTPUT ===")
    # for k, v in OUTPUT_DATA.items():
    #     print(f"{k}:")
    #     for i, item in enumerate(v, 1):
    #         print(f"  {i}. {item}")
    #     print()

if __name__ == "__main__":
    batch_process_images(IMAGE_FOLDER)