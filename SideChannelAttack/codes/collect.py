import time
import json
import os
import signal
import sys
import random
import traceback
import socket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import database
from database import Database

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 1000
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "dataset.json"
MAX_RETRIES = 3
WAIT_BETWEEN_ATTEMPTS = 2  # in seconds

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


"""
Some helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def setup_webdriver():
    """Set up the Selenium WebDriver with Chrome options."""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def retrieve_traces_from_backend(driver):
    """Retrieve traces from the backend API."""
    traces = driver.execute_script("""
        return fetch('/api/get_results')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []

def clear_trace_results(driver, wait):
    """Clear all results from the backend by pressing the button."""
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all results')]")
    clear_button.click()

    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "Cleared"))
    
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

"""
Your implementation starts here.
"""

def collect_single_trace(driver, wait, website_url):
    """ Implement the trace collection logic here. 
    1. Open the fingerprinting website
    2. Click the button to collect trace
    3. Open the target website in a new tab
    4. Interact with the target website (scroll, click, etc.)
    5. Return to the fingerprinting tab and close the target website tab
    6. Wait for the trace to be collected
    7. Return success or failure status
    """
    for attempt in range(MAX_RETRIES):
        try:
            print(f"\n[+] Collecting trace for {website_url}")
            
            # Step 1: Open fingerprinting tool
            driver.get(FINGERPRINTING_URL)
            wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Collect Trace')]")))

            # Step 2: Clear previous traces using helper
            clear_trace_results(driver, wait)

            # Step 3: Click "Collect Trace"
            collect_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Collect Trace')]")
            collect_button.click()

            # Step 4: Open target website in new tab
            driver.execute_script("window.open('', '_blank');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(website_url)

            # Step 5: Interact with the site (e.g., scroll down)
            time.sleep(random.uniform(1, 3))  # Let it load
            for _ in range(random.randint(2, 5)):
                scroll_height = random.randint(300, 1000)
                driver.execute_script(f"window.scrollBy(0, {scroll_height});")
                time.sleep(random.uniform(0.5, 1.5))
            time.sleep(random.uniform(1, 2))

            # Step 6: Close target tab, go back
            driver.close()
            driver.switch_to.window(driver.window_handles[0])

            # Step 7: Wait for trace collection status
            wait.until(EC.text_to_be_present_in_element(
                (By.XPATH, "//div[@role='alert']"),
                "Trace collection complete"
            ))

            # Step 8: Pull trace from backend and return
            traces = retrieve_traces_from_backend(driver)
            if not traces:
                print("[-] No traces returned from backend")
                return None
            return traces[-1]

        except Exception as e:
            print(f"[!] Attempt {attempt + 1} failed for {website_url}: {e}")
            traceback.print_exc()
            if attempt < MAX_RETRIES - 1:
                sleep_time = WAIT_BETWEEN_ATTEMPTS * (attempt + 1)
                print(f"[-] Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    print(f"[X] All attempts failed for {website_url}")
    return None

def collect_fingerprints(driver, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Open the fingerprinting website
    3. Collect traces for each website until the target number is reached
    4. Save the traces to the database
    5. Return the total number of new traces collected
    """
    wait = WebDriverWait(driver, 20)
    new_traces = 0

    if target_counts is None:
        target_counts = {site: TRACES_PER_SITE for site in WEBSITES}

    while not is_collection_complete():
        for idx, site in enumerate(WEBSITES):
            current_count = database.db.get_traces_collected().get(site, 0)
            if current_count >= TRACES_PER_SITE:
                continue  # skip if enough traces collected
            
            print(f"[+] Collecting trace {current_count + 1}/{TRACES_PER_SITE} for {site}")
            trace = collect_single_trace(driver, wait, site)
            if trace:
                saved = database.db.save_trace(site, idx, trace)
                if saved:
                    new_traces += 1
            else:
                print(f"[!] Trace collection failed for {site}")
    return new_traces

def main():
    """ Implement the main function to start the collection process.
    1. Check if the Flask server is running
    2. Initialize the database
    3. Set up the WebDriver
    4. Start the collection process, continuing until the target number of traces is reached
    5. Handle any exceptions and ensure the WebDriver is closed at the end
    6. Export the collected data to a JSON file
    7. Retry if the collection is not complete
    """
    if not is_server_running():
        print("[!] Error: Flask server not running at http://localhost:5000")
        return

    print("[+] Starting fingerprint collection...")
    database.db.init_database()

    driver = setup_webdriver()

    try:
        total_new = collect_fingerprints(driver)
        print(f"[✓] Total new traces collected: {total_new}")
    except Exception as e:
        print(f"[!] Error during collection: {e}")
        traceback.print_exc()
    finally:
        driver.quit()
        database.db.export_to_json(OUTPUT_PATH)

        if not is_collection_complete():
            print("[!] Some traces are still missing. Please re-run the script.")
        else:
            print("[✓] Dataset complete.")

if __name__ == "__main__":
    main()
