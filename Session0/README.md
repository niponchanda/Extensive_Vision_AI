EVA - 7 | Session 0 assignments

1. What are Channels and Kernels (according to EVA)?
Answer: Channels is the container of semantic information. In an image there could be n number of channels. Channel is a feature container.
A kernel is a matrix which convolve with input matrix and extract features. Mostly 3x3 kernel is used, because of its advantages. 
It is also called as feature extractor or filter.

2. Why should we (nearly) always use 3x3 kernels?
Answer: 3x3 kernel have a lot of advantages. 
        >> Less paramerter.
        >> Symmetry
        >> Computing time is less.

4. How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
Answer: 
       1. 199x199 - 3x3 - 197x197
       2. 197x197 - 3x3 - 195x195
       3. 195x195 - 3x3 - 193x193
       4. 193x193 - 3x3 - 191x191
       5. 191x191 - 3x3 - 189x189
       6. 189x189 - 3x3 - 187x187
       7. 187x187 - 3x3 - 185x185
       8. 185x185 - 3x3 - 183x183
       9. 183x183 - 3x3 - 181x181
       10. 181x181 - 3x3 - 179x179
       11. 179x179 - 3x3 - 177x177
       12. 177x177 - 3x3 - 175x175
       13. 175x175 - 3x3 - 173x173
       14. 173x173 - 3x3 - 171x171
       15. 171x171 - 3x3 - 169x169
       16. 169x169 - 3x3 - 167x167
       17. 167x167 - 3x3 - 165x165
       18. 165x165 - 3x3 - 163x163
       19. 163x163 - 3x3 - 161x161
       20. 161x161 - 3x3 - 159x159
       21. 159x159 - 3x3 - 157x157
       22. 157x157 - 3x3 - 155x155
       23. 155x155 - 3x3 - 153x153
       24. 153x153 - 3x3 - 151x151
       25. 151x151 - 3x3 - 149x149
       26. 149x149 - 3x3 - 147x147
       27. 147x147 - 3x3 - 145x145
       28. 145x145 - 3x3 - 143x143
       29. 143x143 - 3x3 - 141x141
       30. 141x141 - 3x3 - 139x139
       31. 139x139 - 3x3 - 137x137
       32. 137x137 - 3x3 - 135x135
       33. 135x135 - 3x3 - 133x133
       34. 133x133 - 3x3 - 131x131
       35. 131x131 - 3x3 - 129x129
       36. 129x129 - 3x3 - 127x127
       37. 127x127 - 3x3 - 125x125
       38. 125x125 - 3x3 - 123x123
       39. 123x123 - 3x3 - 121x121
       40. 121x121 - 3x3 - 119x119
       41. 119x119 - 3x3 - 117x117
       42. 117x117 - 3x3 - 115x115
       43. 115x115 - 3x3 - 113x113
       44. 113x113 - 3x3 - 111x111
       45. 111x111 - 3x3 - 109x109
       46. 109x109 - 3x3 - 107x107
       47. 107x107 - 3x3 - 105x105
       48. 105x105 - 3x3 - 103x103
       49. 103x103 - 3x3 - 101x101
       50. 101x101 - 3x3 - 99x99
       51. 99x99 - 3x3 - 97x97
       52. 97x97 - 3x3 - 95x95
       53. 95x95 - 3x3 - 93x93
       54. 93x93 - 3x3 - 91x91
       55. 91x91 - 3x3 - 89x89
       56. 89x89 - 3x3 - 87x87
       57. 87x87 - 3x3 - 85x85
       58. 85x85 - 3x3 - 83x83
       59. 83x83 - 3x3 - 81x81
       60. 81x81 - 3x3 - 79x79
       61. 79x79 - 3x3 - 77x77
       62. 77x77 - 3x3 - 75x75
       63. 75x75 - 3x3 - 73x73
       64. 73x73 - 3x3 - 71x71
       65. 71x71 - 3x3 - 69x69
       67. 69x69 - 3x3 - 67x67
       68. 67x67 - 3x3 - 65x65
       69. 65x65 - 3x3 - 63x63
       70. 63x63 - 3x3 - 61x61
       71. 61x61 - 3x3 - 59x59
       72. 59x59 - 3x3 - 57x57
       73. 57x57 - 3x3 - 55x55
       74. 55x55 - 3x3 - 53x53
       75. 53x53 - 3x3 - 51x51
       76. 51x51 - 3x3 - 49x49
       77. 49x49 - 3x3 - 47x47
       78. 47x47 - 3x3 - 45x45
       79. 45x45 - 3x3 - 43x43
       80. 43x43 - 3x3 - 41x41
       81. 41x41 - 3x3 - 39x39
       82. 39x39 - 3x3 - 37x37
       83. 37x37 - 3x3 - 35x35
       84. 35x35 - 3x3 - 33x33
       85. 33x33 - 3x3 - 31x31
       86. 31x31 - 3x3 - 29x29
       87. 29x29 - 3x3 - 27x27
       88. 27x27 - 3x3 - 25x25
       89. 25x25 - 3x3 - 23x23
       90. 23x23 - 3x3 - 21x21
       91. 21x21 - 3x3 - 19x19
       92. 19x19 - 3x3 - 17x17
       93. 17x17 - 3x3 - 15x15
       94. 15x15 - 3x3 - 13x13
       95. 13x13 - 3x3 - 11x11
       96. 11x11 - 3x3 - 9x9
       97. 9x9 - 3x3 - 7x7
       95. 7x7 - 3x3 - 5x5
       96. 5x5 - 3x3 - 3x3
       97. 3x3 - 3x3 - 1x1
      

5. How are kernels initialized? 
Answers: Kernels are initilised randomly in the beginning as it is not known what values it should take. While training, back propagation help kernel to get right values.

6. What happens during the training of a DNN?
Answers: While training a deep neural network, the input image matrix of nxn in the batch is feed to the neural network. The kernel is used to extract features
and backpropagation helps in getting the proper values, after that we end output matrix of size 9x9 or 7x7 or any other and check the predicted value which in 
return predicted class name.
