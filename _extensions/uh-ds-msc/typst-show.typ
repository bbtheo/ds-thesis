// typst-show.typ - applies the template to the document

#show: uh-ds-msc.with(
$if(title)$
  title: [$title$],
$endif$
$if(author)$
  author: [$for(author)$$if(author.name)$$author.name$$else$$author$$endif$$sep$, $endfor$],
$endif$
$if(date)$
  date: [$date$],
$endif$
$if(thesis-lang)$
  thesis-lang: "$thesis-lang$",
$endif$
  twoside: $if(twoside)$true$else$false$endif$,
$if(keywords)$
  keywords: ($for(keywords)$"$keywords$"$sep$, $endfor$),
$endif$
$if(classification)$
  classification: [$classification$],
$endif$
$if(depositeplace)$
  depositeplace: "$depositeplace$",
$endif$
$if(additionalinformation)$
  additionalinformation: "$additionalinformation$",
$endif$
$if(abstract)$
  abstract: [$abstract$],
$endif$
$if(quoting)$
  quoting: (text: "$quoting.text$", author: "$quoting.author$"),
$endif$
$if(supervisors)$
  supervisors: ($for(supervisors)$"$supervisors$"$sep$, $endfor$),
$endif$
$if(examiners)$
  examiners: ($for(examiners)$"$examiners$"$sep$, $endfor$),
$endif$
)
