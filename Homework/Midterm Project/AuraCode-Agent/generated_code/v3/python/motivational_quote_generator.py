import random
from datetime import datetime
from typing import List

def get_quotes() -> List[str]:
    """Returns a list of motivational quotes."""
    return [
        "Believe you can and you're halfway there.",
        "The only way to do great work is to love what you do.",
        "Don't watch the clock; do what it does. Keep going.",
        "Everything you've ever wanted is on the other side of fear.",
        "Hardships often prepare ordinary people for an extraordinary destiny.",
        "Your limitation—it's only your imagination.",
        "Push yourself, because no one else is going to do it for you.",
        "Dream it. Wish it. Do it.",
        "Success doesn't just find you. You have to go out and get it.",
        "The harder you work for something, the greater the joy at getting it."
    ]

def generate_border_display(text: str, timestamp: str) -> None:
    """Prints the text and timestamp inside a standard ASCII art border."""
    # We create a list of lines to display. 
    # Note: I removed the leading \n from the quote to keep the border alignment clean.
    lines = [f"Date/Time: {timestamp}", "", text]
    
    # Find the longest line to determine border width
    max_width = max(len(line) for line in lines)
    border_width = max_width + 4
    
    # Use standard ASCII characters to avoid UnicodeEncodeError on Windows consoles
    top_border = "+" + "-" * (border_width - 2) + "+"
    bottom_border = "+" + "-" * (border_width - 2) + "+"
    
    print(top_border)
    for line in lines:
        # Center the text within the border
        padding = border_width - len(line) - 2
        left_pad = padding // 2
        right_pad = padding - left_pad
        print(f"|{' ' * left_pad}{line}{' ' * right_pad}|")
    print(bottom_border)

def main() -> None:
    """Main execution function."""
    quotes = get_quotes()
    selected_quote = random.choice(quotes)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    generate_border_display(selected_quote, current_time)

if __name__ == "__main__":
    main()