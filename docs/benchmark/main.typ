#import "@preview/ilm:1.4.1": *

#set text(lang: "fr", font: "Exo 2")
#show link: set text(fill: blue)
#set list(indent: 1em)
#set enum(indent: 1em)

#show: ilm.with(
  title: [Robot suiveur de personnes],
  author: "Evan Galli, Jilian Lubrat, Eliot Menoret, Antoine-Marie Michelozzi",
  date: datetime(year: 2026, month: 01, day: 22),
  date-format: "[year repr:full]-[month padding:zero]-[day padding:zero]",
  abstract: [
    #text(size:14pt,style: "italic")[\- Optimisation de modèle IA \-]
    #linebreak()#linebreak()
    Un robot autonome capable de reconnaître et de suivre des personnes, reposant sur la reconnaissance de gestes prédéfinis.
    \
    \
    #link("https://github.com/Other-Project/SI5-Autonomous-Intelligent-Systems")[Repo du Projet] --- #link("https://github.com/Other-Project/SI5-Edge-AI")[Repo du Benchmark]
  ],
  //preface: [],
  //bibliography: bibliography("refs.bib"),
  table-of-contents: none,
  figure-index: (enabled: false),
  table-index: (enabled: false),
  listing-index: (enabled: false),
  chapter-pagebreak: true
)

#outline()

#v(2em)

#text(size: 16pt)[*Avant-propos*] \

Des outils d'intelligence artificielle ont été utilisés pour l'amélioration de la rédaction de ce rapport.

#pagebreak()

#set heading(numbering: "I.1.")

#include "context.typ"
#include "benchmarkingCriterias.typ"
#include "devices.typ"
#include "protocol.typ"
#include "pipeline.typ"
#include "benchmark.typ"
#include "conclusion.typ"
