import random
from datetime import datetime
from typing import List

def get_quotes() -> List[str]:
    """Returns a list of motivational quotes."""
    return [
        "Believe you can and you're halfway there.",
        "It always seems impossible until it's done.",
        "The only way to do great work is to love what you do.",
        "Don't watch the clock; do what it does. Keep going.",
        "Your limitation is only your imagination.",
        "Push yourself, because no one else is going to do it for you.",
        "Great things never come from comfort zones.",
        "Dream it. Wish it. Do it."
    ]

def create_border(text: str) -> str:
    """Wraps text in a standard ASCII art border to avoid Unicode encoding errors."""
    lines = text.split('\n')
    max_len = max(len(line) for line in lines)
    
    # Using standard ASCII characters (+, -, |) instead of Unicode box-drawing characters
    top_bottom = "+" + "-" * (max_len + 2) + "+"
    
    bordered_lines = [top_bottom]
    for line in lines:
        bordered_lines.append(f"| {line.ljust(max_len)} |")
    bordered_lines.append(top_bottom)
    
    return "\n".join(bordered_lines)

def main() -> None:
    """Main execution function to display the quote and date."""
    quotes = get_quotes()
    quote = random.choice(quotes)
    date_str = datetime.now().strftime("%A, %B %d, %Y")
    
    content = f"DATE: {date_str}\n\n\"{quote}\""
    print(create_border(content))

if __name__ == '__main__':
    main()