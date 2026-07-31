// University of Helsinki Data Science MSc Thesis Template for Typst/Quarto
// Based on UH_DS_MSc.cls LaTeX template

#let uh-ds-msc(
  title: "Title",
  author: "Author Name",
  date: datetime.today().display("[month repr:long] [day], [year]"),
  thesis-lang: "en",
  twoside: true,
  keywords: (),
  classification: none,
  depositeplace: "",
  additionalinformation: "",
  abstract: none,
  quoting: none,
  supervisors: none,
  examiners: none,
  bibliography-file: none,
  bibliography-style: "apa",
  body
) = {
  // Language-specific strings
  let strings = if thesis-lang == "fi" {
    (
      faculty: "Matemaattis-luonnontieteellinen tiedekunta",
      level: "Pro gradu -tutkielma",
      programme: "Datatieteen maisteriohjelma",
      address: "PL 68 (Pietari Kalmin katu 5)\n00014 Helsingin yliopisto",
      uh: "Helsingin Yliopisto",
      ccs: "ACM Computing Classification System (CCS):",
      abstract-title: "Tiivistelmä",
      toc-title: "Sisällysluettelo",
      bib-title: "Lähteet",
      appendix-title: "Liite",
      list-of-symbols: "Symboliluettelo",
      supervisor-singular: "Ohjaaja:",
      supervisor-plural: "Ohjaajat:",
      examiner-singular: "Tarkastaja:",
      examiner-plural: "Tarkastajat:",
    )
  } else if thesis-lang == "sv" {
    (
      faculty: "Matematisk-naturvetenskapliga fakulteten",
      level: "Magisteravhandling",
      programme: "Magisterprogrammet i data science",
      address: "PB 68 (Pehr Kalms gata 5)\n00014 Helsingfors universitet",
      uh: "Helsingfors Universitet",
      ccs: "ACM Computing Classification System (CCS):",
      abstract-title: "Referat",
      toc-title: "Innehållsförteckning",
      bib-title: "Referenser",
      appendix-title: "Bilaga",
      list-of-symbols: "Nomenklatur",
      supervisor-singular: "Handledare:",
      supervisor-plural: "Handledare:",
      examiner-singular: "Granskare:",
      examiner-plural: "Granskare:",
    )
  } else {
    (
      faculty: "Faculty of Science",
      level: "Master's thesis",
      programme: "Master's Programme in Data Science",
      address: "P. O. Box 68 (Pietari Kalmin katu 5)\n00014 University of Helsinki",
      uh: "University of Helsinki",
      ccs: "ACM Computing Classification System (CCS):",
      abstract-title: "Abstract",
      toc-title: "Table of Contents",
      bib-title: "References",
      appendix-title: "Appendix",
      list-of-symbols: "List of Symbols",
      supervisor-singular: "Supervisor:",
      supervisor-plural: "Supervisors:",
      examiner-singular: "Examiner:",
      examiner-plural: "Examiners:",
    )
  }

  // Page setup
  let binding-offset = if twoside { 0.5cm } else { 0cm }

  // We skip setting document metadata since title/author may be content
  // set document(title: title, author: author)

  set page(
    paper: "a4",
    margin: (
      top: 2.5cm,
      bottom: 2.5cm,
      inside: 2.5cm + binding-offset,
      outside: 2.5cm,
    ),
    numbering: none,
  )

  // Font settings (12pt, similar to LaTeX report class)
  set text(
    font: "New Computer Modern",
    size: 12pt,
    lang: if thesis-lang == "fi" { "fi" } else if thesis-lang == "sv" { "sv" } else { "en" },
  )

  // Paragraph settings
  set par(
    first-line-indent: 1cm,
    justify: true,
    leading: 0.65em * 1.241, // onehalfspacing for 12pt
  )

  // Table cells are too narrow for justified text: it produces ugly
  // mid-word hyphen breaks, so rag right and disable hyphenation there.
  show table: set par(justify: false)
  show table: set text(hyphenate: false)

  // Heading styles
  set heading(numbering: "1.1")

  // Page break before level 1 headings (chapters)
  show heading.where(level: 1): it => {
    colbreak(weak: true)
    v(50pt)
    block[
      #set text(size: 24pt, weight: "bold")
      #if it.numbering != none {
        counter(heading).display()
        [. ]
      }
      #it.body
    ]
    v(40pt)
  }

  show heading.where(level: 2): it => {
    v(1.5em)
    block[
      #set text(size: 14pt, weight: "bold")
      #if it.numbering != none {
        counter(heading).display()
        [ ]
      }
      #it.body
    ]
    v(1em)
  }

  show heading.where(level: 3): it => {
    v(1em)
    block[
      #set text(size: 12pt, weight: "bold")
      #if it.numbering != none {
        counter(heading).display()
        [ ]
      }
      #it.body
    ]
    v(0.5em)
  }

  // Link styling - black for internal links
  show link: set text(fill: rgb("#000000"))
  show ref: set text(fill: rgb("#000000"))

  // Figure caption styling
  show figure.caption: it => {
    set text(size: 10pt, weight: "bold")
    it
  }

  // ============================================
  // TITLE PAGE
  // ============================================
  {
    set page(numbering: none)

    // Use a box to constrain content to single page
    box(height: 100%, width: 100%)[
      #set align(center)

      #v(0.5fr)

      #image("UH-logo.png", width: 5cm)

      #v(0.3cm)

      #text(size: 14pt)[#strings.level]

      #v(0.1cm)

      #text(size: 14pt)[#strings.programme]

      #v(1fr)

      #text(size: 20pt, weight: "bold")[#title]

      #v(1fr)

      #text(size: 14pt)[#author]

      #v(0.5fr)

      #text(size: 14pt)[#date]

      #v(0.5fr)

      // Supervisors and Examiners
      #if supervisors != none or examiners != none [
        #set align(center)
        #{
          // Helper to clean tilde from Pandoc's smart typography
          let clean-name(name) = {
            if type(name) == str {
              name.replace("~", " ")
            } else {
              name
            }
          }
          let sup-label = if supervisors != none {
            if type(supervisors) == array and supervisors.len() > 1 {
              strings.supervisor-plural
            } else {
              strings.supervisor-singular
            }
          } else { "" }
          let exam-label = if examiners != none {
            if type(examiners) == array and examiners.len() > 1 {
              strings.examiner-plural
            } else {
              strings.examiner-singular
            }
          } else { "" }
          grid(
            columns: (auto, auto),
            column-gutter: 1em,
            row-gutter: 0.4em,
            align: (right, left),
            ..if supervisors != none {
              (
                text(size: 11pt)[#sup-label],
                text(size: 11pt)[#if type(supervisors) == array { supervisors.map(s => clean-name(s)).join(", ") } else { clean-name(supervisors) }],
              )
            } else { () },
            ..if examiners != none {
              (
                text(size: 11pt)[#exam-label],
                text(size: 11pt)[#if type(examiners) == array { examiners.map(e => clean-name(e)).join(", ") } else { clean-name(examiners) }],
              )
            } else { () },
          )
        }
      ]

      #v(0.5fr)

      #smallcaps(text(size: 12pt)[#strings.uh])

      #v(0.1cm)

      #text(size: 12pt)[#strings.faculty]

      #v(0.15cm)

      #text(size: 12pt)[#strings.address]

      #v(0.5fr)
    ]

    if twoside {
      pagebreak()
      pagebreak()
    } else {
      pagebreak()
    }
  }

  // ============================================
  // ABSTRACT PAGE
  // ============================================
  if abstract != none {
    set page(numbering: none, margin: (top: 1.5cm, bottom: 1.5cm, left: 2cm, right: 2cm))

    let small-text(content) = text(size: 10pt, content)
    let tiny-text(content) = text(size: 6pt, content)
    let row-height = 30pt
    let small-row-height = 32pt

    // Full page layout
    box(width: 100%, height: 100%)[
      // Header
      #align(center)[
        #small-text[HELSINGIN YLIOPISTO — HELSINGFORS UNIVERSITET — UNIVERSITY OF HELSINKI]
      ]

      #v(1pt)

      // Main form table using Typst table (6 columns for flexible layout)
      #set par(first-line-indent: 0pt)
      #table(
        columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
        // Bottom three rows are auto-sized (min small-row-height would overflow when
        // keywords wrap to multiple lines); the 1fr abstract row absorbs the slack.
        rows: (row-height, row-height, row-height, row-height, 1fr, auto, auto, auto),
        stroke: 0.5pt + black,
        inset: 5pt,
        align: left + top,

        // Row 1: Faculty (half) | Programme (half)
        table.cell(colspan: 3)[
          #tiny-text[Tiedekunta — Fakultet — Faculty]
          #linebreak()
          #small-text[#strings.faculty]
        ],
        table.cell(colspan: 3)[
          #tiny-text[Koulutusohjelma — Utbildningsprogram — Degree programme]
          #linebreak()
          #small-text[#strings.programme]
        ],

        // Row 2: Author (full width)
        table.cell(colspan: 6)[
          #tiny-text[Tekijä — Författare — Author]
          #linebreak()
          #small-text[#author]
        ],

        // Row 3: Title (full width)
        table.cell(colspan: 6)[
          #tiny-text[Työn nimi — Arbetets titel — Title]
          #linebreak()
          #small-text[#title]
        ],

        // Row 4: Level | Date | Pages (each 2 cols)
        table.cell(colspan: 2)[
          #tiny-text[Työn laji — Arbetets art — Level]
          #linebreak()
          #small-text[#strings.level]
        ],
        table.cell(colspan: 2)[
          #tiny-text[Aika — Datum — Month and year]
          #linebreak()
          #small-text[#date]
        ],
        table.cell(colspan: 2)[
          #tiny-text[Sivumäärä — Sidantal — Number of pages]
          #linebreak()
          #small-text[#context counter(page).final().first()]
        ],

        // Row 5: Abstract content (full width, flexible height)
        table.cell(colspan: 6)[
          #tiny-text[Tiivistelmä — Referat — Abstract]
          #v(3pt)
          #small-text[
            #abstract

            #if classification != none [
              #v(0.3cm)
              #strings.ccs #classification
            ]
          ]
        ],

        // Row 6: Keywords (full width)
        table.cell(colspan: 6)[
          #tiny-text[Avainsanat — Nyckelord — Keywords]
          #linebreak()
          #small-text[#keywords.join(", ")]
        ],

        // Row 7: Where deposited (full width)
        table.cell(colspan: 6)[
          #tiny-text[Säilytyspaikka — Förvaringsställe — Where deposited]
          #linebreak()
          #small-text[#depositeplace]
        ],

        // Row 8: Additional information (full width)
        table.cell(colspan: 6)[
          #tiny-text[Muita tietoja — Övriga uppgifter — Additional information]
          #linebreak()
          #small-text[#additionalinformation]
        ],
      )
    ]

    pagebreak()

    if twoside {
      // Empty page for twoside
      pagebreak()
    }
  }

  // ============================================
  // QUOTING PAGE (optional)
  // ============================================
  if quoting != none {
    set page(numbering: none)

    v(1fr)
    align(center)[
      #emph[#quoting.at("text", default: "")]
      #linebreak()
      — #quoting.at("author", default: "")
    ]
    v(1fr)

    pagebreak()
    if twoside {
      pagebreak()
    }
  }

  // ============================================
  // TABLE OF CONTENTS
  // ============================================
  {
    set page(numbering: "i")
    counter(page).update(1)

    heading(outlined: false, numbering: none, level: 1)[#strings.toc-title]

    outline(
      title: none,
      indent: auto,
      depth: 3,
    )

    pagebreak()
    if twoside {
      pagebreak(to: "odd")
    }
  }

  // ============================================
  // MAIN CONTENT
  // ============================================
  {
    set page(numbering: "1")
    counter(page).update(1)

    body
  }
}

// Helper function for appendix formatting
#let appendix(body) = {
  counter(heading).update(0)
  set heading(
    numbering: "A",
    supplement: [Appendix]
  )

  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(20pt)
    block[
      #set text(size: 14pt, weight: "bold")
      Appendix #counter(heading).display(). #it.body
    ]
    v(20pt)
  }

  body
}
