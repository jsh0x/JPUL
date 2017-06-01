__author__ = 'jsh0x'
__version__ = '2.0.0'

import pymssql
import crypt

_k = 462758
_char_map = {1036: '0', 1038: '1', 8732: '2', 7215: '3', 2095: '4', 9270: '5', 4668: '6', 2109: '7', 2116: '8',
	            1096: '9', 2124: 'a', 9303: 'b', 5213: 'c', 5733: 'd', 9319: 'e', 1127: 'f', 1649: 'g', 1137: 'h',
	            1654: 'i', 3702: 'j', 9854: 'k', 1667: 'l', 1166: 'm', 1167: 'n', 9359: 'o', 1171: 'p', 1683: 'q',
	            1175: 'r', 1178: 's', 1692: 't', 1695: 'u', 1184: 'v', 1698: 'w', 1190: 'x', 5802: 'y', 1204: 'z',
	            7865: 'A', 1229: 'B', 1756: 'C', 8413: 'D', 1758: 'E', 1764: 'F', 5869: 'G', 9461: 'H', 1271: 'I',
	            2808: 'J', 1286: 'K', 1801: 'L', 5386: 'M', 1808: 'N', 6931: 'O', 5910: 'P', 1839: 'Q', 4916: 'R',
	            4412: 'S', 1347: 'T', 1353: 'U', 2386: 'V', 4950: 'W', 6490: 'X', 1374: 'Y', 1382: 'Z', 1384: '!',
	            1394: '"', 1919: '#', 7040: '$', 1924: '%', 2439: '&', 3473: "'", 7578: '(', 7585: ')', 1961: '*',
	            5037: '+', 1968: ',', 1461: '-', 1984: '.', 1986: '/', 6596: ':', 7110: ';', 3529: '<', 6097: '=',
	            9690: '>', 7131: '?', 2527: '@', 2021: '[', 1510: '\\', 6632: ']', 1003: '^', 9707: '_', 1517: '`',
	            1516: '{', 6129: '|', 1522: '}', 1523: '~'}
_address = "727825060538191585987860124234755079453535634763314230393770932"
_address = crypt.decrypt(char_map=_char_map, key=_k, value=_address)+"1"
conn = pymssql.connect(server=_address, user='mfg', password='mfg', database='MfgTraveler', login_timeout=10)

def query(cmd:str) -> tuple:
	c = conn.cursor()
	try: c.execute(cmd)
	except: raise ConnectionError
	else: return tuple(c.fetchall())

def modify(cmd:str) -> None:
	c = conn.cursor()
	try: c.execute(cmd)
	except: raise ConnectionError
	else: conn.commit()