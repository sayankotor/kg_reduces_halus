from dewiki_functions import *

#wiki_xml_file = 'F:/simplewiki-20210401/simplewiki-20210401.xml'  # update this
wiki_xml_file = '/home/jovyan/shares/SR004.nfs2/chekalina/PlainTextWikipedia/Untitled Folder/enwiki-20231120-pages-articles-multistream.xml'  # update this
json_save_dir = '/home/jovyan/shares/SR004.nfs2/chekalina/PlainTextWikipedia/big_wikipedia_omonims_new/'

if __name__ == '__main__':
    process_file_text(wiki_xml_file, json_save_dir)