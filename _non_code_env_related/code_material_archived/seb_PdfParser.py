class PdfParser:
    
    def __init__(self, pdf_file):
        
        
        self.pdf = pdf_file
        self.toc = self.__extract_toc__()
        
        
    def __extract_toc__(self):
        toc = -1
        for page in self.pdf:
            tmp = page.get_text('blocks')
    
            for i, block in enumerate(tmp):
                text = block[4].lower().replace('\n', " ").strip()
                
                #if (text == 'table of contents') or (text == 'contents'):
                if ((re.search(r"^\d*\W*\s*table of contents\s*$", text)) \
                    or (re.search(r"^\d*\W*\s*contents\s*$", text)) \
                        or (re.search(r"^\d*\W*\s*index\s*$", text))) \
                    and toc == -1:
                    print('Found ToC on page ' + str(page.number) + ', block ' + str(i))
                    toc = page.number
                    return toc
        
    def extract_sections(self):
        """
        Specific method to extract the full text and Risk Factors section from a pdf.

        Returns
        -------
        toc : int
            Table of Contents page number.
        full_text : str
            Full text of the given pdf.
        risk_section : str
            Risk Factors section text (if available).
        risk_start : int
            Starting page number of the Risk Factors section.
        risk_length : int
            Length of Risk Factors section, in pages.            
        """
        
        toc = self.toc
        risk = 0
        
        risk_starts = []
        full_text = ''
        for page in self.pdf:
            tmp = page.get_text('blocks')
            
            
            for i, block in enumerate(tmp):
                text = block[4].lower().replace('\n', " ").strip()
                full_text = full_text + text
                
                #if (text == 'table of contents') or (text == 'contents'):
                if ((re.search(r"^\d*\W*\s*table of contents\s*$", text)) \
                    or (re.search(r"^\d*\W*\s*contents\s*$", text)) \
                        or (re.search(r"^\d*\W*\s*index\s*$", text))) \
                    and toc == -1:
                    #print('Found ToC on page ' + str(page.number) + ', block ' + str(i))
                    toc = page.number

                #Formatting may vary a lot between prospectuses - this regex seems to catch most of them
                if re.search(r"^\d*\W*\s*risk factors$", text):
                    risk = 1
                    #print('Found Risk section on page ' + str(page.number) + ', block ' + str(i))
                    risk_starts.append(page.number)
                

        if not toc:
            print('Cannot find ToC')
            return 0
        if not risk:
            print('Cannot find Risk')
            return 0
        
        #Lets attempt to extract risk section start/end pages:
        page = self.pdf[toc]
        flag = 0
        for line in page.get_text().split('\n'):
            txt = line.lower().strip()
            if flag and 'risk' not in txt and re.search(r'\w{5,}', txt):
                page_next = re.findall(r'\d+', txt)
                #print('page_next: ' + str(page_next))
                if len(page_next):
                    flag = 0
            if re.search(r"^\d*\W*\s*risk factors[\W\d]*$", txt):#'risk factors' in txt:
                page_num = re.findall(r'\d+', txt)
                flag = 1
                #print('page_num: ' + str(page_num))
    
        if page_num and page_next:
            a = 1
        else:
            flag = 0
            for line in page.get_text('blocks'):
                txt = line[4].lower().replace('\n', ' ').strip()
                if flag and 'risk' not in txt and re.search(r'\w{5,}', txt):
                    page_next = re.findall(r'\d+', txt)
                    #print('page_next: ' + str(page_next))
                    if len(page_next):
                        flag = 0
                if  re.search(r"^\d*\W*\s*risk factors[\W\d]*$", txt):
                    page_num = re.findall(r'\d+', txt)
                    flag = 1
                    #print('page_num: ' + str(page_num))
        
        #For page_next, we pick the last number in the subsequent line
        #The reason for picking the last is that sometimes section names have 
        #numbers in the, fx "... regarding the 6.5 notes..." or similar.
        pages = int(page_next[-1]) - int(page_num[0])
        
        if len(risk_starts) == 1:
            risk_start = risk_starts[0]
            #print('Risk section must start on page ' + str(risk_start) + ' and end on page ' + str(risk_start + pages))
        else:
            #risk_start = max([risk for risk in risk_starts if int(page_num[0]) <= risk + 1])
            risk_start = min([risk for risk in risk_starts if risk > toc])
            #print('Risk section start ambiguous, best guess is pages ' + str(risk_start) + ' to ' + str(risk_start + pages))
            
        #We now have the desired pages. Lets read them 
        risk_section = ' '
        for idx in range(risk_start, risk_start + pages):
            risk_section = risk_section + self.pdf[idx].get_text()
        
        self.toc = toc
        return toc, full_text, risk_section, risk_start, risk_start + pages
    
    def extract_full_text(self):
        """
        Extract full text from a pdf.

        Returns
        -------
        full_text : str
            Full text from pdf.

        """
        
        full_text = ''
        for page in self.pdf:
            text = page.get_text().encode('utf8').decode('utf8').replace('\n', ' ')
            
            full_text = full_text + text
                
        return full_text
    
    
    def extract_section_by_title(self, section):
        """
        Attempt to extract the contents of a section by its title.

        Parameters
        ----------
        section : str
            Title of section. Does not need to be an exact match.

        Returns
        -------
        section_text : str
            Extracted text.

        """
        
        
        section = section.lower()
        flag = 0
        for page in self.pdf:
            tmp = page.get_text('blocks')
    
            for i, block in enumerate(tmp):
                text = block[4].lower().replace('\n', " ").strip()
                
                if re.search(r"^\d*\W*\s*{}\s*$".format(section), text):
                    print('Found section ' + section +  ' on page ' + str(page.number))
                    flag = 1
                    num = page.number
                    title = section
                    
        if not flag:
            print('No exact match found for section. Attempting to find similarly titled section?')
            
            #Save best match so far
            ratio = 0
            num = 0
            
            for page in self.pdf:
                tmp = page.get_text('blocks')
        
                for i, block in enumerate(tmp):
                    text = block[4].lower().replace('\n', " ").strip()
                    
                   
                    if lev.ratio(text, section) > 0.75 and lev.ratio(text, section) > ratio:
                        print('Found similar title: ' + text + ' on page ' + str(page.number))
                        flag = 1
                        num = page.number
                        title = text
                        
        
        
        if flag:
            #Lets attempt to extract section start/end pages:
            flag = 0
            exact_flag = 0
            
            #If we cannot find exact match, let us instead keep the best one.
            ratio = 0 
            
            #ToC may be multiple pages
            #Lets assume they are at most 3 pages?
            for i in range(self.toc, self.toc + 3):
                
                page = self.pdf[i]
                for line in page.get_text().split('\n'):
                    txt = line.lower().strip()
                    #The below line could perhaps be improved to a fuzzy string matching, like we did above
                    if exact_flag and section not in txt and re.search(r'\w{5,}', txt):
                        page_next = re.findall(r'\d+', txt)
                        #print('page_next: ' + str(page_next))
                        if len(page_next):
                            exact_flag = 0
                    else:
                        if flag and section not in txt and re.search(r'\w{5,}', txt):
                            page_next = re.findall(r'\d+', txt)
                            #print('page_next: ' + str(page_next))
                            if len(page_next):
                                flag = 0
                    
                    if re.search(rf"^\d*\W*\s*{title}[\W\d]*$", txt):
                        page_num = re.findall(r'\d+', txt)
                        exact_flag = 1
                        #print('Found exact match: page_num: ' + str(page_num))
                        #break
                    else:
                        if lev.ratio(text, section) > 0.75 and lev.ratio(text, section) > ratio:
                            page_num = re.findall(r'\d+', txt)
                            flag = 1
                            #print('Found better fuzzy match: page_num: ' + str(page_num))
            
            
            pages = int(page_next[-1]) - int(page_num[0])
            
            #We now have the desired pages. Lets read them 
            section_text = ' '
            for idx in range(num, num + pages):
                section_text = section_text + self.pdf[idx].get_text()
                
            return section_text
            
        else:
            print('Unable to find section ' + section)
            return None