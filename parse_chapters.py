import xml.etree.ElementTree as ET

def parse_chapters(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get book title
    book_title = root.get('title')
    print(f"\nBook: {book_title}\n")
    
    # Find all chapter elements
    chapters = root.findall('chapter')
    
    for chapter in chapters:
        # Get chapter info
        chapter_id = chapter.get('id')
        chapter_title = chapter.get('title')
        
        # Get chapter text and limit to first 100 chars
        chapter_text = chapter.text.strip() if chapter.text else ""
        # cut off top line and strip
        chapter_text = chapter_text[chapter_text.find("\n") + 1:].strip()
        
        preview = chapter_text[:100] + "..." if len(chapter_text) > 100 else chapter_text
        
        print(f"=== {chapter_title} ({chapter_id}) ===")
        print(f"{preview}\n")

if __name__ == "__main__":
    xml_path = "texts/processed/dorian_grey.xml"
    parse_chapters(xml_path)
