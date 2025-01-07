# import re
# import os
# from xml.etree import ElementTree as ET
# from xml.dom import minidom

# def process_dorian_grey():
#     # Create processed directory if it doesn't exist
#     os.makedirs('texts/processed', exist_ok=True)
    
#     # Read the file
#     with open('texts/dorian_grey.txt', 'r', encoding='utf-8') as f:
#         text = f.read()
    
#     # Create root XML element
#     root = ET.Element("book")
#     root.set("title", "The Picture of Dorian Gray")
    
#     # Split into chapters using regex
#     # Look for chapter markers and keep them with the content
#     chapter_pattern = r'(CHAPTER [IVXLC\d]+\..*?)(?=CHAPTER [IVXLC\d]+\.|$)'
#     chapters = re.findall(chapter_pattern, text, re.DOTALL)
    
#     # Process chapters
#     for i, content in enumerate(chapters):
#         # Create chapter element
#         chapter = ET.SubElement(root, "chapter")
#         chapter.set("id", f"chapter_{i}")
#         chapter.set("title", f"Chapter {i}")
#         chapter.text = content.strip()
    
#     # Pretty print XML
#     xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    
#     # Save as XML
#     output_path = 'texts/processed/dorian_grey.xml'
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(xml_str)
    
#     print(f"Processed and saved to {output_path}")

# def process_time_machine():
#     # Create processed directory if it doesn't exist
#     os.makedirs('texts/processed', exist_ok=True)
    
#     # Read the file
#     with open('texts/time_machine.txt', 'r', encoding='utf-8') as f:
#         text = f.read()
    
#     # Create root XML element
#     root = ET.Element("book")
#     root.set("title", "The Time Machine")
    
#     # Split into chapters using 4 or more newlines as separator
#     chapters = re.split(r'\n{4,}', text)
    
#     # Track actual chapter number (no skipping)
#     chapter_num = 1
    
#     # Process chapters
#     for content in chapters:
#         if content.strip():  # Only process non-empty chapters
#             # Create chapter element
#             chapter = ET.SubElement(root, "chapter")
#             chapter.set("id", f"chapter_{chapter_num-1}")  # Keep 0-based ids
#             chapter.set("title", f"Chapter {chapter_num}")
#             chapter.text = content.strip()
#             chapter_num += 1
    
#     # Pretty print XML
#     xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    
#     # Save as XML
#     output_path = 'texts/processed/time_machine.xml'
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(xml_str)
    
#     print(f"Processed and saved to {output_path}")

# if __name__ == "__main__":
#     process_time_machine()
