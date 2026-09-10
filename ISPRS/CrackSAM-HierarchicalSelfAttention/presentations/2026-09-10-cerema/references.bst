% BibTeX declarations for the defence's citation macros, without a bibliography
% slide. Full author lists remain in references.bib; six or more use et al.
% Requires \DeclareRef, \DeclareMyRef and hyperref (see citations.tex).

ENTRY
  { author title booktitle shortbooktitle journal year url reflabel ownwork }
  {} {}

STRINGS { names result venue }
INTEGERS { name.index name.count }

FUNCTION {field.or.null}
{ duplicate$ empty$ { pop$ "" } 'skip$ if$ }

FUNCTION {format.authors}
{ author 'names :=
  names num.names$ 'name.count :=
  names #1 "{f{.~}.~~}{vv~}{ll}{, jj}" format.name$ 'result :=
  name.count #5 >
    { result " \emph{et~al.}" * }
    { #2 'name.index :=
      { name.index name.count > #0 = }
      { result ", " *
        names name.index "{f{.~}.~~}{vv~}{ll}{, jj}" format.name$ *
        'result :=
        name.index #1 + 'name.index :=
      }
      while$
      result
    }
  if$
}

FUNCTION {format.title}
{ url empty$
    { title }
    { "\href{" url * "}{" * title * "}" * }
  if$
}

FUNCTION {format.venue}
{ shortbooktitle empty$
    { booktitle empty$ { journal field.or.null } { booktitle } if$ }
    { shortbooktitle }
  if$
  'venue :=
  venue empty$
    'skip$
    { ownwork field.or.null "yes" =
        { venue write$ }
        { "\textit{" venue * "}" * write$ }
      if$
      ", " write$
    }
  if$
}

FUNCTION {output.entry}
{ ownwork field.or.null "yes" =
    { "\DeclareMyRef{" }
    { "\DeclareRef{" }
  if$
  cite$ * "}{" * reflabel * "}{" * write$
  format.authors write$
  ", « " write$
  format.title write$
  " », " write$
  format.venue
  year write$ ".}" write$ newline$
}

FUNCTION {inproceedings} { output.entry }
FUNCTION {article} { output.entry }
FUNCTION {misc} { output.entry }
FUNCTION {default.type} { output.entry }

READ
ITERATE {call.type$}
