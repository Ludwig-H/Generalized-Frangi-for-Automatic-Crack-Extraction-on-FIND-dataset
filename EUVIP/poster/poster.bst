% Compact numeric BibTeX style for this poster; use \bibliographystyle{poster}.
% Citation order, first author + et al., unchanged title case, linked titles.
% Requires hyperref. DOI takes precedence over URL; neither is printed in full.
% The label-width pass follows the standard BibTeX styles by Patashnik/Trickey.

ENTRY
  { author editor organization title journal booktitle shortbooktitle volume number pages
    year publisher edition howpublished note doi url eprint archivePrefix }
  {} { label }

STRINGS { body s t longest.label }
INTEGERS { number.label longest.label.width }

FUNCTION {field.or.null}
{ duplicate$ empty$ { pop$ "" } 'skip$ if$ }

FUNCTION {emphasize}
{ duplicate$ empty$ { pop$ "" } { "\emph{" swap$ * "}" * } if$ }

FUNCTION {format.names}
{ 's :=
  s #1 "{f.~}{vv~}{ll}{, jj}" format.name$
  s num.names$ #1 > { " et~al." * } 'skip$ if$
}

FUNCTION {format.authors}
{ author empty$
    { editor empty$
        { organization field.or.null }
        { editor format.names ", ed." * }
      if$
    }
    { author format.names }
  if$
}

FUNCTION {format.title}
{ title empty$
    { "Missing title in " cite$ * warning$ "[Untitled]" }
    { title }
  if$
  't :=
  doi empty$
    { url empty$ { t } { "\href{" url * "}{" * t * "}" * } if$ }
    { "\href{https://doi.org/" doi * "}{" * t * "}" * }
  if$
}

% Receive the formatted title; start the entry and an empty metadata list.
FUNCTION {output.header}
{ newline$ "\bibitem{" cite$ * "}" * write$ newline$
  format.authors duplicate$ empty$
    { pop$ }
    { write$ ", " write$ }
  if$
  add.period$ write$
  "" 'body :=
}

FUNCTION {append.part}
{ 's :=
  s empty$ 'skip$
    { body empty$ { s } { body ", " * s * } if$ 'body := }
  if$
}

FUNCTION {format.volume}
{ volume field.or.null
  number empty$ 'skip$
    { duplicate$ empty$
        { pop$ "no.~" number * }
        { "(" * number * ")" * }
      if$
    }
  if$
}

% Accept either single hyphens or TeX double hyphens in page ranges.
FUNCTION {n.dashify}
{ 't := ""
    { t empty$ #0 = }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" =
            { { t #1 #1 substring$ "-" = }
              { "-" * t #2 global.max$ substring$ 't := }
              while$
            }
            { "--" * t #2 global.max$ substring$ 't := }
          if$
        }
        { t #1 #1 substring$ * t #2 global.max$ substring$ 't := }
      if$
    }
  while$
}

FUNCTION {format.pages}
{ pages empty$ { "" } { "pp.~" pages n.dashify * } if$ }

FUNCTION {fin.entry}
{ year field.or.null append.part
  body empty$ 'skip$ { " " write$ body add.period$ write$ } if$
  newline$
}

FUNCTION {article}
{ format.title output.header
  journal field.or.null emphasize append.part
  format.volume append.part
  pages field.or.null n.dashify append.part
  fin.entry
}

FUNCTION {inproceedings}
{ format.title output.header
  shortbooktitle empty$ { booktitle field.or.null } { shortbooktitle } if$
  emphasize append.part
  format.pages append.part
  fin.entry
}

FUNCTION {misc}
{ format.title output.header
  howpublished field.or.null append.part
  eprint empty$ 'skip$
    { archivePrefix empty$ { "arXiv" } { archivePrefix } if$
      ":" * eprint * append.part
    }
  if$
  publisher field.or.null append.part
  note field.or.null append.part
  fin.entry
}

FUNCTION {book}
{ format.title emphasize output.header
  publisher field.or.null append.part
  edition empty$ 'skip$ { edition " ed." * append.part } if$
  fin.entry
}

FUNCTION {conference} { inproceedings }
FUNCTION {proceedings} { book }
FUNCTION {unpublished} { misc }
FUNCTION {default.type} { misc }

READ

FUNCTION {initialize.labels}
{ "" 'longest.label := #1 'number.label := #0 'longest.label.width := }

FUNCTION {measure.label}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label := label width$ 'longest.label.width := }
    'skip$
  if$
}

EXECUTE {initialize.labels}
ITERATE {measure.label}

FUNCTION {begin.bib}
{ preamble$ empty$ 'skip$ { preamble$ write$ newline$ } if$
  "\begin{thebibliography}{" longest.label * "}" * write$ newline$
}

EXECUTE {begin.bib}
ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }

EXECUTE {end.bib}
