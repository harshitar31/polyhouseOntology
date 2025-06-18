import json
import re
import random
from typing import List, Dict, Tuple

class SPARQLQuestionRephraser:
    def __init__(self):
        # Define rephrasing patterns for different question types
        self.rephrase_patterns = {
            # "What is" questions
            "what_is": [
                "What is {subject}?",
                "Can you tell me what {subject} is?",
                "What does {subject} refer to?",
                "I need to know what {subject} is.",
                "Could you explain what {subject} is?",
                "What exactly is {subject}?",
                "Define {subject}.",
                "Describe {subject}."
            ],
            
            # "What type" questions
            "what_type": [
                "What type of {object} is {subject}?",
                "What kind of {object} is {subject}?",
                "What category of {object} does {subject} belong to?",
                "Can you identify the type of {object} that {subject} is?",
                "What classification of {object} is {subject}?",
                "Which type of {object} is {subject}?",
                "What sort of {object} is {subject}?"
            ],
            
            # "What are" questions
            "what_are": [
                "What are {subject}?",
                "Can you list {subject}?",
                "What {subject} are there?",
                "Which {subject} exist?",
                "Tell me about {subject}.",
                "Identify {subject}.",
                "What {subject} can you find?"
            ],
            
            # Value/measurement questions
            "what_value": [
                "What is the {property} of {subject}?",
                "What {property} does {subject} have?",
                "Can you tell me the {property} of {subject}?",
                "What is {subject}'s {property}?",
                "I need the {property} of {subject}.",
                "Find the {property} of {subject}.",
                "Get the {property} for {subject}.",
                "Retrieve the {property} of {subject}."
            ]
        }
    
    def extract_question_components(self, question: str) -> Dict[str, str]:
        """Extract key components from the question for rephrasing"""
        components = {}
        
        # Pattern for "What is the X of Y?"
        value_pattern = r"What is the (\w+) of (.+?)\?"
        match = re.search(value_pattern, question)
        if match:
            components['type'] = 'what_value'
            components['property'] = match.group(1)
            components['subject'] = match.group(2)
            return components
        
        # Pattern for "What type of X is Y?"
        type_pattern = r"What type of (\w+) is (.+?)\?"
        match = re.search(type_pattern, question)
        if match:
            components['type'] = 'what_type'
            components['object'] = match.group(1)
            components['subject'] = match.group(2)
            return components
        
        # Pattern for "What are the X that Y?"
        features_pattern = r"What are the (\w+) that (.+?)\?"
        match = re.search(features_pattern, question)
        if match:
            components['type'] = 'what_are'
            components['subject'] = match.group(1)
            return components
        
        # Pattern for "What is X?"
        simple_pattern = r"What is (.+?)\?"
        match = re.search(simple_pattern, question)
        if match:
            components['type'] = 'what_is'
            components['subject'] = match.group(1)
            return components
        
        return components
    
    def generate_rephrased_questions(self, original_question: str, num_variants: int = 3) -> List[str]:
        """Generate rephrased versions of the original question"""
        components = self.extract_question_components(original_question)
        
        if not components or 'type' not in components:
            # If we can't parse the question, create simple variations
            return self.create_simple_variations(original_question, num_variants)
        
        question_type = components['type']
        patterns = self.rephrase_patterns.get(question_type, [])
        
        if not patterns:
            return self.create_simple_variations(original_question, num_variants)
        
        rephrased = []
        selected_patterns = random.sample(patterns, min(num_variants, len(patterns)))
        
        for pattern in selected_patterns:
            try:
                rephrased_question = pattern.format(**components)
                rephrased.append(rephrased_question)
            except KeyError:
                # If formatting fails, skip this pattern
                continue
        
        # Fill remaining slots with simple variations if needed
        while len(rephrased) < num_variants:
            simple_vars = self.create_simple_variations(original_question, 1)
            if simple_vars and simple_vars[0] not in rephrased:
                rephrased.extend(simple_vars)
            else:
                break
        
        return rephrased[:num_variants]
    
    def create_simple_variations(self, question: str, num_variants: int) -> List[str]:
        """Create simple variations when pattern matching fails"""
        variations = []
        
        # Simple rephrasing without prefixes
        if "What is" in question:
            base = question.replace("What is", "").replace("?", "").strip()
            variations.extend([
                f"Tell me about {base}.",
                f"Describe {base}.",
                f"Explain {base}."
            ])
        elif "What are" in question:
            base = question.replace("What are", "").replace("?", "").strip()
            variations.extend([
                f"List {base}.",
                f"Show me {base}.",
                f"Find {base}."
            ])
        else:
            # Generic variations
            variations.extend([
                question.replace("What", "Which"),
                question.replace("?", " exactly?"),
                question.replace("What", "Can you find what")
            ])
        
        return variations[:num_variants]
    
    def process_jsonl_file(self, input_file: str, output_file: str, variants_per_question: int = 3):
        """Process JSONL file and generate rephrased questions"""
        original_data = []
        rephrased_data = []
        
        try:
            # Read original data
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        original_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                        continue
            
            print(f"Successfully loaded {len(original_data)} questions from {input_file}")
            
        except FileNotFoundError:
            print(f"Error: File '{input_file}' not found. Please check the file path.")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        
        # Generate rephrased versions
        for i, item in enumerate(original_data, 1):
            original_prompt = item['prompt']
            completion = item['completion']
            
            # Generate rephrased questions
            rephrased_questions = self.generate_rephrased_questions(
                original_prompt, variants_per_question
            )
            
            # Add rephrased versions to the dataset
            for rephrased_question in rephrased_questions:
                rephrased_item = {
                    'prompt': rephrased_question,
                    'completion': completion
                }
                rephrased_data.append(rephrased_item)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"Processed {i}/{len(original_data)} questions...")
        
        # Combine original and rephrased data
        all_data = original_data + rephrased_data
        
        # Write to output file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in all_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"\n✅ Processing complete!")
            print(f"Original questions: {len(original_data)}")
            print(f"Rephrased questions: {len(rephrased_data)}")
            print(f"Total questions: {len(all_data)}")
            print(f"Data saved to: {output_file}")
            
        except Exception as e:
            print(f"Error writing output file: {e}")

# Usage function for your actual data
def process_your_data(input_filename: str, output_filename: str = None, variants: int = 3):
    """
    Process your actual JSONL file
    
    Args:
        input_filename: Path to your input JSONL file
        output_filename: Path for output file (optional, defaults to input_expanded.jsonl)
        variants: Number of variants per question (default: 3)
    """
    if output_filename is None:
        base_name = input_filename.rsplit('.', 1)[0]
        output_filename = f"{base_name}_expanded.jsonl"
    
    rephraser = SPARQLQuestionRephraser()
    rephraser.process_jsonl_file(input_filename, output_filename, variants)
    
    return output_filename

# Example usage with your actual data
def main():
    """
    Example usage - replace 'your_data.jsonl' with your actual file path
    """
    
    # Process your actual data file
    input_file = "D:/VolumeEStuff/sony/polyhouseOntology/newdataset.jsonl"  # Replace with your actual file path
    output_file = "D:/VolumeEStuff/sony/moreQuestions.jsonl"
    
    print("SPARQL Question Rephraser")
    print("=" * 40)
    
    # You can change the number of variants per question here
    variants_per_question = 3
    
    # Process the file
    rephraser = SPARQLQuestionRephraser()
    rephraser.process_jsonl_file(input_file, output_file, variants_per_question)

# Quick function to use with your file
def expand_my_dataset(file_path: str, num_variants: int = 3):
    """
    Quick function to expand your dataset
    
    Usage:
    expand_my_dataset("path/to/your/file.jsonl", 3)
    """
    return process_your_data(file_path, variants=num_variants)

if __name__ == "__main__":
    main()