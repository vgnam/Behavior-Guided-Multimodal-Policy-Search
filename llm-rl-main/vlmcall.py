import litellm
from litellm import completion

# ====== CONFIG ======
FPT_API_KEY = "sk-uDXg03MCrYREykzUKG0g2kHPZFjDmIvAShRkL1q0dCdohnxf"
MODEL_NAME = "openai/DeepSeek-V3.2-Speciale"
API_BASE = "https://mkp-api.fptcloud.com/v1"

# ====== SET GLOBAL CONFIG ======
litellm.api_key = FPT_API_KEY
litellm.api_base = API_BASE

# ====== TEST CALL ======
def main():
    try:
        response = completion(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            temperature=0.7,
            max_tokens=100,
            timeout=60,
        )

        print("===== RESPONSE =====")
        print(response["choices"][0]["message"]["content"])

    except Exception as e:
        print("❌ ERROR:")
        print(e)


if __name__ == "__main__":
    main()