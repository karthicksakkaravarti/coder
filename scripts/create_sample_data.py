import pandas as pd
import os

def create_sample_data(output_path, num_samples=1000):
    data = []
    for i in range(num_samples):
        if i % 4 == 0:
            instruction = f"Create a heading saying 'Heading {i}'"
            html_code = f"<h1>Heading {i}</h1>"
        elif i % 4 == 1:
            instruction = f"Create a paragraph about topic {i}"
            html_code = f"<p>This is a paragraph about topic {i}. It contains some sample text to demonstrate HTML generation.</p>"
        elif i % 4 == 2:
            instruction = f"Create a link to example.com with text 'Link {i}'"
            html_code = f'<a href="https://example.com">Link {i}</a>'
        else:
            instruction = f"Create an unordered list with items A, B, and C for list {i}"
            html_code = f"<ul><li>Item A for list {i}</li><li>Item B for list {i}</li><li>Item C for list {i}</li></ul>"
        
        data.append({"instruction": instruction, "html_code": html_code})
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample data created at {output_path}")

if __name__ == "__main__":
    create_sample_data('data/processed/html_dataset.csv')