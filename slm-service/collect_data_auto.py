"""
Automated data collection script for threshold training.
Runs fill-in-the-blank prompts automatically without manual input.
"""
import sys
from pathlib import Path
import asyncio

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from main import generate_response
from data_collector import DataCollector
import argparse

# Sentence completion prompts for data collection
SENTENCE_COMPLETION_PROMPTS = [
    # Science & Nature
    "The capital of France is ",
    "Water is composed of two hydrogen atoms and one ",
    "The speed of light in a vacuum is approximately ",
    "Photosynthesis converts carbon dioxide and water into glucose and ",
    "The largest planet in our solar system is ",
    "DNA stands for ",
    "The process by which plants make food is called ",
    "The smallest unit of matter is an ",
    "The force that pulls objects toward Earth is called ",
    "The chemical symbol for gold is ",
    
    # Geography
    "The longest river in the world is the ",
    "Mount Everest is located in the ",
    "The largest ocean on Earth is the ",
    "The country with the most people is ",
    "The Sahara Desert is located in ",
    "The Great Wall of China was built to protect against ",
    "The Amazon rainforest is primarily located in ",
    "The smallest country in the world is ",
    "The country known as the Land of the Rising Sun is ",
    "The capital of Australia is ",
    
    # History
    "World War II ended in the year ",
    "The first man to walk on the moon was ",
    "The Renaissance period began in ",
    "The Industrial Revolution started in ",
    "The fall of the Berlin Wall happened in ",
    "The American Civil War was fought between ",
    "The first successful airplane flight was made by the ",
    "The ancient city of Rome was founded in the year ",
    "The Magna Carta was signed in the year ",
    "The French Revolution began in the year ",
    
    # Technology & Computing
    "The first computer was called ",
    "The programming language Python was named after ",
    "The World Wide Web was invented by ",
    "A byte consists of ",
    "The operating system developed by Microsoft is called ",
    "The company that created the iPhone is ",
    "HTML stands for ",
    "The first email was sent in the year ",
    "The largest social media platform is ",
    "The inventor of the telephone was ",
    
    # Literature & Arts
    "Shakespeare wrote the play ",
    "The author of '1984' is ",
    "The Mona Lisa was painted by ",
    "The novel 'To Kill a Mockingbird' was written by ",
    "Beethoven was a famous ",
    "The painting 'Starry Night' was created by ",
    "The author of 'Pride and Prejudice' is ",
    "The musical instrument with 88 keys is a ",
    "The author of 'The Great Gatsby' is ",
    "The famous play 'Romeo and Juliet' was written by ",
    
    # Mathematics
    "The value of pi is approximately ",
    "A triangle with three equal sides is called an ",
    "The square root of 16 is ",
    "The number of degrees in a circle is ",
    "The mathematical constant e is approximately ",
    "A polygon with eight sides is called an ",
    "The sum of angles in a triangle is ",
    "The number 2 raised to the power of 10 equals ",
    "The derivative of x squared is ",
    "The area of a circle is calculated using the formula ",
    
    # General Knowledge
    "The human body has ",
    "The largest organ in the human body is the ",
    "The process of cell division is called ",
    "The currency of Japan is the ",
    "The largest mammal in the world is the ",
    "The smallest mammal in the world is the ",
    "The fastest land animal is the ",
    "The tallest mountain in the world is ",
    "The deepest ocean trench is the ",
    "The largest continent is ",
    
    # Sentence completion variations
    "The opposite of hot is ",
    "A group of lions is called a ",
    "The study of weather is called ",
    "The process of turning liquid into gas is called ",
    "The hardest natural substance on Earth is ",
    "The planet closest to the Sun is ",
    "The largest moon in our solar system is ",
    "The inventor of the light bulb was ",
    "The first successful vaccine was for the disease ",
    "When it's cold outside, people wear ",
    
    # Context-based completions
    "To make coffee, you need to ",
    "In a library, books are organized by ",
    "A doctor uses a ",
    "To travel across the ocean, you would use a ",
    "The main ingredient in bread is ",
    "To send a letter, you need a ",
    "A person who studies stars is called an ",
    "The tool used to measure temperature is a ",
    "To communicate over long distances, people use ",
    "The oldest city in Eastern Europe is ",
    
    # Academic subjects
    "The study of living organisms is called ",
    "The study of matter and energy is called ",
    "The study of the past is called ",
    "The study of human behavior is called ",
    "The study of numbers and shapes is called ",
    "The study of language structure is called ",
    "The study of the Earth's physical features is called ",
    "The study of government and politics is called ",
    "The study of chemical reactions is called ",
    "The study of the mind and mental processes is called ",
    
    # More sentence completions
    "The first person to reach the South Pole was ",
    "The largest desert in the world is the ",
    "The smallest planet in our solar system is ",
    "The process of converting sunlight into electricity is called ",
    "The most abundant gas in Earth's atmosphere is ",
    "The longest bone in the human body is the ",
    "The capital of Japan is ",
    "The largest lake in the world is ",
    "The first element in the periodic table is ",
    "The speed of sound in air is approximately ",
]

def main():
    parser = argparse.ArgumentParser(description='Automated data collection for threshold training')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='Maximum tokens to generate per prompt (default: 50)')
    parser.add_argument('--data-file', type=str, default=None,
                        help='File to save training data (default: slm-service/training_data.jsonl)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='LLM service host (default: 127.0.0.1, use A100 IP if running on A100)')
    parser.add_argument('--port', type=int, default=8081,
                        help='LLM service port (default: 8081)')
    parser.add_argument('--num-prompts', type=int, default=None,
                        help='Number of prompts to run (default: all prompts)')
    parser.add_argument('--use-chat-template', action='store_true',
                        help='Use chat template formatting')
    args = parser.parse_args()
    
    # Initialize data collector
    data_collector = DataCollector(data_file=args.data_file)
    print(f"📊 Automated data collection starting...")
    print(f"   Data will be saved to: {data_collector.data_file}")
    print(f"   Current data points: {data_collector.get_data_count()}")
    print(f"   Threshold is set to 0.0 (all tokens will be transmitted)")
    print(f"   Connecting to LLM service at {args.host}:{args.port}")
    print(f"   Total prompts to run: {args.num_prompts or len(SENTENCE_COMPLETION_PROMPTS)}")
    print("-" * 60)
    
    # Select prompts to run
    prompts_to_run = SENTENCE_COMPLETION_PROMPTS[:args.num_prompts] if args.num_prompts else SENTENCE_COMPLETION_PROMPTS
    
    successful = 0
    failed = 0
    
    for i, prompt in enumerate(prompts_to_run, 1):
        print(f"\n[{i}/{len(prompts_to_run)}] Processing: {prompt}")
        try:
            asyncio.run(generate_response(
                prompt,
                max_tokens=args.max_tokens,
                use_chat_template=args.use_chat_template,
                simulate_network=False,
                data_collector=data_collector,
                host=args.host,
                port=args.port
            ))
            successful += 1
            print(f"   ✓ Success. Total data points: {data_collector.get_data_count()}")
        except Exception as e:
            failed += 1
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print(f"📊 Data collection complete!")
    print(f"   Successful prompts: {successful}")
    print(f"   Failed prompts: {failed}")
    print(f"   Total data points collected: {data_collector.get_data_count()}")
    print(f"   Data saved to: {data_collector.data_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()

