# latexmkrc
$latex = 'uplatex -synctex=1 -halt-on-error -interaction=nonstopmode -file-line-error %O %S';
$bibtex = 'upbibtex %O %B';
$biber = 'biber --bblencoding=utf8 -u -U --output_safechars %O %B';
$makeindex = 'upmendex %O -o %D %S';
$dvipdf = 'dvipdfmx %O -o %D %S';
$pdf_mode = 3;
$max_repeat = 5;
