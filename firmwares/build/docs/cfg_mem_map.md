### Application "QS Portable Radio CPS" reads following areas of memory:

| Address| Length    | Content (probably)  |
| :---:  |  ---:     | :------  |
|`0x0E70`| `0xA8`    | Common settings + DTMF settings |
|`0x0D60`| `0xCF`    | MR Channels Parameters (format unknown) |
|`0x0F50`| `0x0C80`  | MR Channels names, 16 bytes/ch |
|`0x0000`| `0x0D60`  | MR Channels frequency, 16 bytes/ch * 200 + VFO Channels⁺, 16 bytes/ch * 14 |
|`0x1C00`| `0x0100`  | DTMF Contacts, 16 bytes/contact * 16 |  
|`0x0E40`| `0x28`    | FM channels⁺, 2 bytes/ch * 20 |
|`0x0F18`| `0x8`     | ? |

<hr>

### VCO Channels (ranges) settings `0x0C80` - `0x0D5F`
| Freq range | address    | Lenght  |
| :---       |  :---      | :------ |
| F1 - A     |  `0x0C80`  |  `0xF`  |
| F1 - B     |  `0x0C90`  |  `0xF`  |
| F2 - A     |  `0x0CA0`  |  `0xF`  |
| F2 - B     |  `0x0CB0`  |  `0xF`  |
| F3 - A     |  `0x0CC0`  |  `0xF`  |
| F3 - B     |  `0x0CD0`  |  `0xF`  |
| F4 - A     |  `0x0CE0`  |  `0xF`  |
| F4 - B     |  `0x0CF0`  |  `0xF`  |
| F5 - A     |  `0x0D00`  |  `0xF`  |
| F5 - B     |  `0x0D10`  |  `0xF`  |
| F6 - A     |  `0x0D20`  |  `0xF`  |
| F6 - B     |  `0x0D30`  |  `0xF`  |
| F7 - A     |  `0x0D40`  |  `0xF`  |
| F7 - B     |  `0x0D50`  |  `0xF`  |

#### Single range format (16 bytes)
| Offset  |  Lenght  | Info |
| :---    |  :---    | :------      |
| `0x0`   |  4       | Rx frequency |
| `0x4`   |  4       | Tx frequency delta (0 means Rx=Tx) |
| `0x8`   |  3       | ?? |
| `0xB`   |  1       | Some flags, Bit4 = AM on |
| `0xC`   |  4       | ?? |
<hr>

### Additional addresses found by trial and error:

| Address| Length    | Content (probably)  | Position in menu |
| :---:  |  ---:     | :------  | --: |
|`0x0E7D`| `1`       | Backlight : `0`=Off, `1..5`=Seconds | 16 |
|`0x0E97`| `1`       | Power on display mode: `0`=Fullscreen, `1`=Welcome info, `2`=Voltage | 45 |
|`0x0E98`| `4`       | Power on password saved as uint32_le, for example `3f420f00` = `999999`| - |
|`0x0EA0`| `1`       | Voice Prompt (OFF=0, CHI=1, ENG=2) | 21 |
|`0x0EB0`| `0x10`    | `Quansheng` string (welcome screen)| - |
|`0x0EC0`| `0x10`    | `UV-K5` string (welcome screen, 2 line)| - |
|`0x0F40`| `1`       | F-LOCK: `0`=Off, `1`=FCC, `2`=CE, `3`=GB, `4`=430, `5`=438 | 53 |
|`0x0F41`| `1`       | 350TX: `01`=On, `00`=Off | 52 |
|`0x0F42`| `1`       | Radio remotely "Killed" by DTMF "Kill code" `01`=Killed, `00`=Normal mode | - |
|`0x0F43`| `1`       | 200TX: `01`=On, `00`=Off | 54 |
|`0x0F44`| `1`       | 500TX: `01`=On, `00`=Off | 55 |
|`0x0F45`| `1`       | 350EN: `01`=On, `00`=Off | 56 |
|`0x0F46`| `1`       | SCREN (Scrambling Enabled): `01`=On, `00`=Off | 57 |
|`0x0EE0`| `8`       | ANI DTMF ID | 35 |
|`0x0EE8`| `8`       | Kill Code ||
|`0x0EF0`| `8`       | Revive Code ||
|`0x0EF8`| `8`       | DTMF Up Code | 36 |
|`0x0F08`| `8`       | DTMF Down Code | 37 |


⁺ Side note: all frequencies are stored as follows: 446.05625MHz is just 0x02A8A0B9 (44605625 in hex) with the bytes reversed (so the EEPROM contains B9 A0 A8 02.) For the FM channels, only 3 digits are used and the zeroes are cut, e.g. 91.1 FM becomes 911, 38F in hex, stored as 8F 03.


### Battery calibration area

| Address  | Length    | Content           | Sample value |
| :---:    |  ---:     | :------           | --           |
| `0x1F40` |  `2`      | Battery Level 0   | 1258         |
| `0x1F42` |  `2`      | Battery Level 1   | 1747         |
| `0x1F44` |  `2`      | Battery Level 2   | 1876         |
| `0x1F46` |  `2`      | Battery Level 3   | 1901         |
| `0x1F48` |  `2`      | Battery Level 4   | 2004         |
| `0x1F4a` |  `2`      | Battery Level 5   | 2300         |

Remarks:
 * Sample value is RAW data from ADC
 * Voltage value is calculated with formula: `Volts` = `7.6`× `RAW_ADC` / `BattLevel3`. For example for ADC=2000, Voltage = 7.6×2000/1901 ≈ 7.99V
 * if during boot `BattLevel0` will be larger than `5000` then these default levels will be set: `BattLevel0`=1900 and `BattLevel1`=2000
 * `BattLevel5`is always set up to 2300 and value from EEPROM is ignored


### Function keys

| Address  | Length    | Content                     | Range        |
| :---:    |  :---:    | :------                     | --           |
| `0x0E91` |  `1`      | Key 1 - short press action  | `0` - `8`    |
| `0x0E92` |  `1`      | Key 1 - long  press action  | `0` - `8`    |
| `0x0E93` |  `1`      | Key 2 - short press action  | `0` - `8`    |
| `0x0E94` |  `1`      | Key 2 - long  press action  | `0` - `8`    |

#### Actions

| Value  | Action             |
| :---:  | :---               |
| `0`    | No action          |
| `1`    | Flashlight         |
| `2`    | Power select       |
| `3`    | Monitor            |
| `4`    | Scan on/off        |
| `5`    | VOX on/off         |
| `6`    | Alarm on/off       |
| `7`    | FM Radio on/off    |
| `8`    | Transmitting 1750  |
