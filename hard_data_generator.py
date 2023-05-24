def generate():
    szum_0 = [20816, 52294]
    hard_example_0 = [63, 1501, 1512, 5298, 6967, 6970, 7597, 9455, 9552, 9687, 9717, 10005, 10025, 10237, 10259,
                      11584, 11848, 12103, 12184, 12321, 13184, 13408, 13719, 14857, 16921, 17518, 18102, 18324, 18832,
                      19360, 21608, 21766, 21960, 23104, 25386, 26728, 28242, 28357, 28628, 29311, 29341, 29345, 29560,
                      29713, 31661, 31683, 32682, 33378, 36560, 37059, 41358, 42063, 44955, 46726, 46860, 48638, 49463,
                      50390, 50808, 50930, 51017, 51019, 54371, 56454, 57486, 57686, 59784]

    szum_1 = [3532, 3811, 4129, 4502, 5842, 6328, 8904, 9550, 14756, 16130, 19782, 20672, 21990, 24424, 25910, 26376,
              29062, 29320, 29434, 30508, 31134, 32070, 34010, 34640, 35236, 37038, 40644, 40976, 41266, 50572, 50632]
    hard_example_1 = [67, 1080, 1259, 1287, 2202, 2823, 2831, 4066, 4715, 6226, 6885, 7440, 7599, 8689, 9221, 9256,
                      10205,
                      10239, 10667, 10842, 12791, 12805, 12932, 17209, 17706, 17802, 18196, 18293, 19894, 21601, 21791,
                      21944, 23656, 26444, 31366, 35360, 36141, 42460, 47928, 48862, 51544, 52624, 54576, 54752, 59354]

    szum_2 = [2322, 4476, 4935, 4955, 5013, 5095, 5103, 5129, 39074, 39582]
    hard_example_2 = [178, 448, 493, 563, 584, 657, 1047, 2792, 2847, 2875, 2877, 2887, 4355, 4414, 4428, 4576, 4986,
                      5065, 5110, 5162, 5216, 5602, 5624, 5656, 5722, 5790, 7005, 7146, 7196, 7732, 7920, 8154, 8226,
                      8470, 8480, 8670, 8752, 9098, 9472, 11600, 11655, 11711, 11737, 12196, 13735, 14468, 15278, 15434,
                      15660, 16030, 17448, 18423, 18684, 18776, 18864, 19423, 19502, 19568, 19808, 21382, 21798, 21880,
                      22062, 22083, 22554, 22884, 23194, 23770, 23794, 24198, 24997, 25148, 25220, 25412, 25994, 26624,
                      26626, 27258, 27614, 28078, 28200, 28234, 28672, 28786, 30881, 31185, 31197, 32475, 33088, 33906,
                      34444, 35882, 36522, 36982, 37403, 37501, 37533, 37601, 37974, 38068, 39421, 39734, 40076, 41290,
                      41382, 41802, 42532, 42934, 43575, 44274, 44424, 44898, 45332, 45631, 46214, 47759, 47767, 47912,
                      48070, 48354, 48460, 48811, 48837, 48915, 48933, 48975, 48991, 49009, 49021, 49067, 49070, 49264,
                      49282, 50576, 50840, 52006, 52358, 52582, 52677, 52834, 52938, 53264, 54756, 54762, 54834, 54854,
                      56222, 56569, 57956, 59720]

    szum_3 = [7080, 10994, 16678, 19215, 50340, 50417, 51686, 52085]
    hard_example_3 = [1097, 5332, 6347, 7584, 10091, 10984, 11781, 12183, 14692, 16676, 16748, 22204, 22643, 24059,
                      25034, 30130, 31596, 31962, 32276, 32642, 43783, 46246, 47217, 54782, 55116, 55194]

    szum_4 = [24798, 29206, 59915]
    hard_example_4 = [1248, 3290, 3370, 4564, 5673, 8207, 9046, 10918, 11024, 12438, 12870, 13650, 14260, 14335, 17592,
                      17870, 18504, 20287, 20337, 22747, 23486, 23956, 24292, 27514, 27738, 29340, 29922, 32512, 32776,
                      33888, 35324, 37074, 37794, 38301, 38822, 39304, 39375, 42078, 43068, 45048, 47558, 49196, 53912,
                      53990, 55630, 56452, 58105, 59753, 59759, 59766]

    szum_5 = [2554, 4450, 4625, 5904, 6018, 6418, 7270, 9450, 9568, 10804, 10831, 14512, 15975, 16011, 16092, 16698,
              22779, 23252, 24504, 25678, 26842, 28116, 30082, 35310, 37680, 37836, 39427, 40824, 52914, 53063, 53638,
              56200, 56224, 57302, 57510, 57662]
    hard_example_5 = [132, 8731, 12157, 12181, 12692, 20976, 26017, 26398, 27296, 27502, 28654, 28770, 31287, 31301,
                      31413, 31415, 32323, 32445, 32507, 38234, 38287, 38512, 38553, 38592, 38698, 40654, 41072, 42428,
                      43574, 51795, 52210, 55442, 59701, 59726, 59731, 59747]

    szum_6 = [1435, 3229, 5084, 9534, 10089, 13067, 14487, 15450, 26983, 34059, 38259, 42729, 44728, 49026, 49960,
              51508]
    hard_example_6 = [1269, 2148, 5684, 15862, 17728, 22561, 26940, 28302, 30770, 31606, 31806, 34520, 38152, 38185,
                      38187, 39697, 39709, 42450, 42571, 42723, 43592, 44732, 45580, 48507, 50086, 50856, 59928]

    szum_7 = [212, 934, 1019, 1982, 2576, 2844, 4680, 4774, 4794, 5416, 5976, 6221, 6241, 6259, 6278, 6315, 6319, 6333,
              6339, 6361, 6375, 6381, 6397, 6403, 11786, 11789, 12000, 12970, 17703, 17882, 18046, 18834, 18844, 21693,
              21701, 22090, 22984, 23387, 23451, 23507, 25125, 25868, 26049, 27912, 28620, 29238, 33582, 33612, 37834,
              38672, 39357, 39586, 39659, 40193, 40211, 40239, 40311, 40357, 40387, 46248, 46274, 46462, 46770, 47203,
              47325, 47595, 48228, 49644, 49829, 54207, 54623, 54678, 54917, 56034, 59724]
    hard_example_7 = [140, 340, 1075, 2014, 2292, 5068, 7437, 17086, 19534, 20048, 23140, 24938, 24990, 25300, 26882,
                      28014, 28632, 28670, 30202, 32236, 33656, 34132, 35731, 37900, 38650, 39038, 40684, 41054, 42606,
                      43471, 43618, 43686, 43998, 44590, 44853, 44865, 45806, 47560, 48874, 50994, 53242, 53501, 53866,
                      54194, 54212, 54218, 57235, 57255, 57357, 58094]

    szum_8 = [2901, 9608, 35246, 39378, 39395, 39419]
    hard_example_8 = [3570, 4148, 6066, 6506, 7606, 7909, 8029, 8031, 8033, 8210, 8730, 12112, 12317, 12522, 12936,
                      15963,
                      19124, 22320, 23582, 30994, 32573, 33038, 34328, 34758, 35232, 40466, 41218, 42112, 42384, 42832,
                      49548, 50505, 51576, 52932, 53854, 54082, 56180, 56286, 56866, 56978, 57042]

    szum_9 = [80, 902, 1612, 2764, 5704, 5718, 5740, 5806, 5868, 5898, 6092, 6272, 7264, 10944, 11196, 11416, 13002,
              13666,
              14547, 14579, 14582, 14762, 14796, 14884, 14886, 15072, 15728, 17817, 18382, 18405, 20918, 24006, 24613,
              28262, 28279, 28422, 30049, 30184, 32342, 35464, 36282, 36606, 45654, 45810, 46298, 46316, 49904, 50724,
              51280, 51942, 53216, 54036, 55240, 55730, 56152]
    hard_example_9 = [1826, 38230, 39448, 41270, 41334, 45528, 45917, 49143, 49163, 58539]

    szum = szum_0 + szum_1 + szum_2 + szum_3 + szum_4 + szum_5 + szum_6 + szum_7 + szum_8 + szum_9
    hard_examples = hard_example_0 + hard_example_1 + hard_example_2 + hard_example_3 + hard_example_4 + hard_example_5 \
                    + hard_example_6 + hard_example_7 + hard_example_8 + hard_example_9
    all = szum + hard_examples

    return szum, hard_examples, all