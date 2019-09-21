question format taken from this [book](https://drive.google.com/file/d/1xxmZhGVYPtLwNMqKH4o8dIynPDlmB-Wk/view?usp=sharing)
EXAMPLE 1-11 Density, Mass, Volume
A 47.3-mL sample of ethyl alcohol (ethanol) has a mass of 37.32 g. What is its density?

here the line just after the EXAMPLE is showing the labels from which topic this questions is from.
we can add this label to the the particular question if we want.
question is starting with EXAMPLE and ending with "?, ."

What should be regular expression for this.
  pattern = "^EXAMPLE.*.\[.?]$" 
  this is the pattern but I have to remove the newline after the EXAMPLE ending line then it will match all the questions that I want.
  

or from Exercises.

text is
  https://pastebin.com/NmEKyjS8
  
 with regular expression "^[0-9]{1,2}.*[/\r?\n|\r/g].*.*\?$"
 
 we can extract the questions
 ```text
 in text format

2. What was Mendeleev’s contribution to the construction of
the modern periodic table?
24. What is the difference between ionization and dissociation
in aqueous solution?
35. Some chemical reactions reach an equilibrium, rather than
going to completion. What is “equal” in such an equilibrium?
15. Define (a) acids, (b) bases, (c) salts, and (d) molecular compounds.
16. How can a salt be related to a particular acid and a particular base?
48. The air we inhale contains O2. We exhale CO2 and H2O.
Does this suggest that our bodily processes involve oxidation? Why?
57. Which of the following would displace hydrogen when a
piece of the metal is dropped into dilute H2SO4 solution?
62. Arrange the metals listed in Exercise 61 in order of increasing activity.
63. What is the order of decreasing activity of the halogens?
64. Of the possible displacement reactions shown, which
one(s) could occur?
100. Identify the decomposition reactions.
101. (a) Do any of these reactions fit into more than one class?
132. How many moles of oxygen can be obtained by the decomposition of 10.0 grams of reactant in each of the following
reactions?
134. What mass of Zn is needed to displace 20.6 grams of Cu
from CuSO4  5H2O?
 ```
 
 ```text
 or in csv format
 
match,group,is_participating,start,end,content
1,0,yes,75,161,2. What was Mendeleev’s contribution to the construction of
the modern periodic table?
2,0,yes,5582,5665,24. What is the difference between ionization and dissociation
in aqueous solution?
3,0,yes,7066,7188,"35. Some chemical reactions reach an equilibrium, rather than
going to completion. What is “equal” in such an equilibrium?"
4,0,yes,8350,8496,"15. Define (a) acids, (b) bases, (c) salts, and (d) molecular compounds.
16. How can a salt be related to a particular acid and a particular base?"
5,0,yes,11406,11531,48. The air we inhale contains O2. We exhale CO2 and H2O.
Does this suggest that our bodily processes involve oxidation? Why?
6,0,yes,13576,13691,57. Which of the following would displace hydrogen when a
piece of the metal is dropped into dilute H2SO4 solution?
7,0,yes,14269,14408,62. Arrange the metals listed in Exercise 61 in order of increasing activity.
63. What is the order of decreasing activity of the halogens?
8,0,yes,14409,14484,"64. Of the possible displacement reactions shown, which
one(s) could occur?"
9,0,yes,21977,22084,100. Identify the decomposition reactions.
101. (a) Do any of these reactions fit into more than one class?
10,0,yes,29147,29275,132. How many moles of oxygen can be obtained by the decomposition of 10.0 grams of reactant in each of the following
reactions?
11,0,yes,29696,29774,134. What mass of Zn is needed to displace 20.6 grams of Cu
from CuSO4  5H2O?

 ```
 
 ```json
 [
  [
    {
      "content": "2. What was Mendeleev’s contribution to the construction of\nthe modern periodic table?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 75,
      "endPos": 161
    }
  ],
  [
    {
      "content": "24. What is the difference between ionization and dissociation\nin aqueous solution?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 5582,
      "endPos": 5665
    }
  ],
  [
    {
      "content": "35. Some chemical reactions reach an equilibrium, rather than\ngoing to completion. What is “equal” in such an equilibrium?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 7066,
      "endPos": 7188
    }
  ],
  [
    {
      "content": "15. Define (a) acids, (b) bases, (c) salts, and (d) molecular compounds.\n16. How can a salt be related to a particular acid and a particular base?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 8350,
      "endPos": 8496
    }
  ],
  [
    {
      "content": "48. The air we inhale contains O2. We exhale CO2 and H2O.\nDoes this suggest that our bodily processes involve oxidation? Why?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 11406,
      "endPos": 11531
    }
  ],
  [
    {
      "content": "57. Which of the following would displace hydrogen when a\npiece of the metal is dropped into dilute H2SO4 solution?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 13576,
      "endPos": 13691
    }
  ],
  [
    {
      "content": "62. Arrange the metals listed in Exercise 61 in order of increasing activity.\n63. What is the order of decreasing activity of the halogens?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 14269,
      "endPos": 14408
    }
  ],
  [
    {
      "content": "64. Of the possible displacement reactions shown, which\none(s) could occur?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 14409,
      "endPos": 14484
    }
  ],
  [
    {
      "content": "100. Identify the decomposition reactions.\n101. (a) Do any of these reactions fit into more than one class?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 21977,
      "endPos": 22084
    }
  ],
  [
    {
      "content": "132. How many moles of oxygen can be obtained by the decomposition of 10.0 grams of reactant in each of the following\nreactions?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 29147,
      "endPos": 29275
    }
  ],
  [
    {
      "content": "134. What mass of Zn is needed to displace 20.6 grams of Cu\nfrom CuSO4 \u0006 5H2O?",
      "isParticipating": true,
      "groupNum": 0,
      "groupName": null,
      "startPos": 29696,
      "endPos": 29774
    }
  ]
]
 ```
