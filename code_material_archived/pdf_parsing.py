
import pymupdf # imports the pymupdf library
import pymupdf4llm
import json
import re


# part to convert pdf to markdown
# doc = pymupdf.open("./risk_factors.pdf") # open a document
# for page in doc: # iterate the document pages
#   text = page.get_text() # get plain text encoded as UTF-8
#print(text) # print the text

md_text = pymupdf4llm.to_markdown("./Prospectus-2007-04-25.pdf") # convert the document to markdown

# write markdown string to some file
output = open("./out-markdown.md", "w")
output.write(md_text)
output.close()


# part to convert markdown to json
def read_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.rstrip('\n') for line in lines]

def merge_same_format_lines(lines):
    merged_lines = []
    prev_format = None
    current_line = ''

    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_bold = line.startswith('**') and line.endswith('**')
        is_italic = line.startswith('_') and line.endswith('_')
        is_bold_italic = line.startswith('**_') and line.endswith('_**')
        if is_bold_italic:
            line_format = 'bold_italic'
        elif is_bold:
            line_format = 'bold'
        elif is_italic:
            line_format = 'italic'
        else:
            line_format = 'normal'

        line_content = line.strip('*_')
        if prev_format == line_format:
            current_line += ' ' + line_content
        else:
            if current_line:
                formatted_line = format_line(prev_format, current_line)
                merged_lines.append(formatted_line)
            current_line = line_content
            prev_format = line_format

    if current_line:
        formatted_line = format_line(prev_format, current_line)
        merged_lines.append(formatted_line)

    return merged_lines

def format_line(line_format, content):
    if line_format == 'bold':
        return f'**{content}**'
    elif line_format == 'bold_italic':
        return f'**_{content}_**'
    elif line_format == 'italic':
        return f'_{content}_'
    else:
        return content

def build_hierarchy(merged_lines):
    hierarchy = []
    current_title = None
    current_subtitle = None
    current_subsubtitle = None
    current_body = []

    for line in merged_lines:
        line = line.strip()
        if not line:
            continue

        is_bold = line.startswith('**') and line.endswith('**')
        is_bold_italic = line.startswith('**_') and line.endswith('_**')
        line_content = line.strip('*_')
        is_upper = line_content.isupper()

        if is_bold_italic:
            # Subsubtitle
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_subsubtitle = line_content
            current_body = []
            # Add subsubtitle to hierarchy
            for section in hierarchy:
                if section['title'] == current_title:
                    for subsection in section['subsections']:
                        if subsection['subtitle'] == current_subtitle:
                            if 'subsubsections' not in subsection:
                                subsection['subsubsections'] = []
                            subsection['subsubsections'].append({'subsubtitle': current_subsubtitle, 'body': ''})
                            break
                    break
        elif is_bold and is_upper:
            # Title
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_title = line_content
            current_subtitle = None
            current_subsubtitle = None
            current_body = []
            hierarchy.append({'title': current_title, 'subsections': []})
        elif is_bold:
            # Subtitle
            add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
            current_subtitle = line_content
            current_subsubtitle = None
            current_body = []
            # Add subtitle to hierarchy
            for section in hierarchy:
                if section['title'] == current_title:
                    section['subsections'].append({'subtitle': current_subtitle, 'body': '', 'subsubsections': []})
                    break
        else:
            # Body text
            current_body.append(line)

    add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle)
    return hierarchy

def add_body_to_hierarchy(current_body, hierarchy, current_title, current_subtitle, current_subsubtitle):
    body_text = ' '.join(current_body).strip()
    if not body_text:
        return
    if current_subsubtitle:
        # Add body to current subsubtitle
        for section in hierarchy:
            if section['title'] == current_title:
                for subsection in section['subsections']:
                    if subsection['subtitle'] == current_subtitle:
                        for subsub in subsection['subsubsections']:
                            if subsub['subsubtitle'] == current_subsubtitle:
                                subsub['body'] += ' ' + body_text
                                return
    elif current_subtitle:
        # Add body to current subtitle
        for section in hierarchy:
            if section['title'] == current_title:
                for subsection in section['subsections']:
                    if subsection['subtitle'] == current_subtitle:
                        subsection['body'] += ' ' + body_text
                        return
    elif current_title:
        # Add body to current title
        for section in hierarchy:
            if section['title'] == current_title:
                if 'body' not in section:
                    section['body'] = ''
                section['body'] += ' ' + body_text
                return

def save_hierarchy(hierarchy, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=4, ensure_ascii=False)

def main():
    markdown_file = 'out-markdown.md'  # Replace with your markdown file path
    output_file = 'output.json'        # Output JSON file path

    lines = read_markdown(markdown_file)
    merged_lines = merge_same_format_lines(lines)
    hierarchy = build_hierarchy(merged_lines)
    save_hierarchy(hierarchy, output_file)
    print(f'Hierarchy saved to {output_file}')

if __name__ == '__main__':
    main() 