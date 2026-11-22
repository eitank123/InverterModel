# BMS notes: board marked "ZB16ST-C4-V8.5"

Status: **not identified yet.** No datasheet, manual, protocol or config software has been found for this
exact board. Everything below is labeled: **[photo]** means read from the owner's photo,
**[inferred]** means my reading of the layout, not a confirmed fact, and **[source]** gives a URL.

Research date: 2026-11-22.

## 1. What the photo shows [photo]

Photo: `/root/.claude/uploads/723ab5c1-df1d-5a03-8859-37d1d3ebbfc8/ddec9590-image.jpg` (EXIF-rotated, 3000x4000).

| Item | Observation |
|---|---|
| Silkscreen | `ZB16ST-C4-V8.5`, `ZLYF-C1S-20190618-01`, `NO.1004000000303`, `RoHS`, a "B" in a rounded square inside a circle, with a small ® next to it |
| Construction | It looks like **two boards stacked** [inferred]. A black control board (the upper half is under kapton tape) sits on top of a white power board. The white board carries the labeled pins `DFET`, `RSENSE`, `CFET`, `PSENSE`, a 3-pin header labeled `ISP`, a mounting hole, and two large SMD power parts (MOSFETs or a shunt). The labeled pins appear to be the control-to-power interconnect: discharge FET gate drive, current-sense shunt, charge FET gate drive, and pack sense [inferred]. |
| Balance connector | One long single-row connector carrying 16 white leads plus 1 black lead, so **17 positions = B0 (B-) to B16 = 16S** [inferred from the wire count]. The wire jacket text reads like "AWM 1007 ... XINHENG" (a cable maker, not the BMS brand). |
| Comm/aux connector | Left edge, black housing with latch. **8 pins are visible** in the close-up, not 6, with a black multi-wire harness. Purpose unknown: UART, RS485, CAN, display or key switch are all possible [inferred]. |
| 2-pin connector | White 2-pin (looks like JST-PH/XH style) above the comm connector, probably NTC or a switch [inferred]. |
| Debug | 4-pad row labeled `VDD ... GND` next to the MCU, likely SWD/ISP (VDD, SWDIO/data, SWCLK/clock, GND) [inferred]. There is a separate 3-pin `ISP` header on the power board. |
| ICs | MCU: LQFP-64 (16 pins per side). Upper section: one QFP (left, partly hidden) and one TSSOP/SSOP (about 2x15 to 2x22 pins), probably the AFE [inferred]. **Markings are not readable** in the photo because of the kapton tape, glare and focus. |
| Power supply | DC-DC buck with an inductor marked "220" (22 uH). |
| Indicator | One SMD LED near the comm connector. |
| Power leads | Thick red and black leads, in heat-shrink. |

**Chemistry and voltage:** 16S suggests either 48/51.2 V LiFePO4 or 60 V (67.2 V full) NMC. The board does
not show which. The firmware thresholds decide, so **do not assume LFP or NMC** until you read the settings
or the seller's label. The stacked control-board-on-power-board design, with an MCU and ISP, looks like an
OEM pack BMS (e-moped or swappable battery type) rather than a retail DIY BMS [inferred].

## 2. Identification attempts (all negative)

Exact-string web searches found **zero relevant hits** for `ZB16ST`, `ZB16ST-C4`, `ZB16ST-C4-V8.5`,
`ZLYF-C1S`, `ZLYF-C1S-20190618`, sibling guesses (`ZB10ST`, `ZB13ST`, `ZB14ST`, `ZB20ST`), the Chinese
queries `ZB16ST 保护板 16串` and `丝印 ZB16ST`, and the silkscreen nets `DFET/CFET/RSENSE/PSENSE + ISP`.
A GitHub code search for `ZB16ST` and `ZLYF` returned only unrelated matches. The mydigit.cn thread about
an unknown-string-count board (https://www.mydigit.cn/thread-536166-1-1.html) was blocked by its firewall,
and its search snippet was not this board.

"ZLYF-C1S-20190618-01" reads like a PCB fab or design code with a date (2019-06-18) [inferred].
"Ziggy C1S" (a 60 V e-moped, https://ziggywheels.com/c1s/) also uses "C1S" and a 60 V pack, but **there
is no evidence** that it is linked to this board. Treat it as a coincidence unless the pack came from such
a vehicle.

### Candidate families (none confirmed)

| Candidate | For | Against | Confidence |
|---|---|---|---|
| JBD / Xiaoxiang (Jiabaida) | Very common 16S smart BMS with UART/RS485 and the well-documented `DD A5 ... 77` protocol | JBD retail boards use `SP16S0xx`, `UP16S0xx`, `ZP16S011` style codes (e.g. https://store.elyfevspare.com/product/nmc-16s-40a-zp16s011-jbd-bms-223e) and a single PCB, not `ZB16ST`. No JBD logo seen. | low |
| Daly | Common 16S, UART/RS485/CAN, many OEM variants | Daly boards are usually potted or in an aluminum case with "DL16S"-style labels (https://manuals.plus/ae/1005003578486911); a stacked open board is atypical | low |
| ANT / JK / Seplos / Pace | 16S smart BMS with MCU | Their silkscreen and form factor differ, and none uses "ZB...ST" naming | very low |
| Unknown OEM pack BMS (e-moped / swappable battery) | Stacked control + power boards, labeled interconnect pins, ISP, 8-pin harness, no retail footprint online | — | **most likely category**, but no brand identified |

The "B"-in-a-circle logo with ® is a registered brand mark, but I could not match it to a BMS maker.

## 3. Specs, thresholds, protocol: unknown

No spec sheet was found. The following are **not known** for this board: continuous or peak current,
OVP/UVP/OCP/SCP/temperature thresholds, balancing current, sleep current, comm-port pinout, baud rate and
register map. Do not use numbers from other brands' sheets as if they apply here.

Read them off the hardware instead:
* **Current class:** count the MOSFETs on the power board and read their part numbers. Read the shunt
  value (the RSENSE resistors, e.g. "R001" = 1 mOhm).
* **Chemistry:** after power-up, if you can talk to it, read the cell OVP setting. About 3.6-3.75 V means LFP
  and about 4.2-4.25 V means NMC (typical ranges for these chemistries, not this board's values).

## 4. Protocol and software: how to find out

If it turns out to be a known protocol, these open-source projects cover the common Chinese BMS families.
I could not reach github.com from this session, so **the links were not verified today**. They are
well-known projects as of my knowledge:
* JBD/Xiaoxiang UART/RS485/BLE (`DD A5 03 00 FF FD 77` basic-info request, 9600 8N1):
  https://github.com/syssi/esphome-jbd-bms
* ANT: https://github.com/syssi/esphome-ant-bms · JK: https://github.com/syssi/esphome-jk-bms
* Daly UART: built into ESPHome (`daly_bms`, https://esphome.io/components/sensor/daly_bms.html)
* Multi-brand Victron/Venus OS driver: https://github.com/Louisvdw/dbus-serialbattery (now maintained in a
  fork by mr-manuel)
* Multi-brand BLE for Home Assistant: https://github.com/fl4p/batmon-ha

**Passive identification plan (no transmit needed first):**
1. Photograph the 8-pin connector pin side and trace each pin to a chip on the board. An RS485 transceiver
   (SOIC-8, e.g. "MAX485/SP3485/THVD"-type marking) means A/B lines. A CAN transceiver (SOIC-8,
   "TJA1050/1042/SIT1050") means CANH/CANL. Direct MCU traces with series resistors suggest TTL UART.
2. Use a multimeter (pack connected, see section 5) to find GND, a supply pin (often 3.3 V, 5 V or 12 V)
   and idle-high lines (TTL TX idles at 3.3 V; RS485 A/B rest a few hundred mV apart).
3. Connect a **3.3 V USB-TTL adapter** (RX only) or a **USB-RS485** or **USB-CAN** adapter, as identified,
   **to listen**. Try 9600, 19200 and 115200 baud, 8N1. For CAN, try 250 k and 500 k.
4. Only then send a known query, e.g. the JBD read `DD A5 03 00 FF FD 77` at 9600 baud, and the Daly
   `A5 40 90 08 ...` frame, and see whether it answers.
5. If nothing answers, the protocol is proprietary. With no reply, the remaining options are firmware
   extraction over SWD (the VDD/GND pads) or getting the vendor's tool from whoever sold the pack.

## 5. Safe bring-up (general practice, not from a manual for this board)

General order used by smart-BMS makers (e.g. JBD, Daly and Seplos manuals; see the Seplos 16S spec,
https://cdn.shopify.com/s/files/1/0623/5389/9720/files/SEPLOS_16S_LFP_100A_BMS_SPECIFICATION_V16.pdf):
1. **Before connecting anything to the BMS**, wire the 17-lead harness to the cells and measure every
   lead with the plug unplugged. Pin 1 (black) is B- = 0 V, and each next pin should be about one cell
   higher, up to B16 = full pack voltage. The plug's pin order must match. Check the polarity mark (the
   arrow drawn on the connector in the photo).
2. Connect the **B- power lead** (thick black) to the pack negative first.
3. Plug in the **balance connector** (the harness is already wired, so the leads land all at once,
   lowest-first by design).
4. Connect B+ if the board has a separate sense or power lead (the thick red one here may be B+ or P+,
   so **trace it before connecting**).
5. Connect the load or charger to P-/P+ (or C-) last.
6. **Wake-up [inferred]:** many smart BMS ship asleep with the FETs off and wake on a charger connected to
   P+/P- for a moment, a key/switch input (the 2-pin connector is a candidate), or comm activity. Try a
   current-limited charger or bench supply at pack voltage first.
7. Disconnect in the reverse order. Never plug the balance connector into a board whose B- is not
   connected, and never hot-swap it partly seated.

Use a current-limited supply or a fuse for first power. With no thresholds known, watch cell voltages
yourself during the first charge.

## 6. What would settle the identification

* Clear, glare-free close-ups (kapton tape lifted) of **every IC marking**: the MCU (LQFP-64), the
  TSSOP/QFP in the upper section (AFE: e.g. BQ76940/BQ76952, SH367309, ML5238 or similar, unconfirmed),
  any SOIC-8 transceivers, and the MOSFET and shunt markings on the power board.
* The **back side** of both boards, and any label or sticker on the pack, the enclosure, or the comm
  harness's far end (display, vehicle connector).
* The source of the pack (vehicle or brand). An OEM BMS is usually documented only by the pack vendor.
* Then search the AFE and MCU numbers plus "16S" on 21ic / mydigit / endless-sphere, and send the photo and
  markings to the seller.

---

## Update: BGR Drive search (2026-11-22, read-only)

No file in the BGR shared drive mentions "ZB16ST", "ZLYF", or the DFET/CFET/PSENSE labels.
The closest lead is a Shenzhen Smart Li-ion Energy Technology Co. (smart-bms.com) BMS that the team
already used. Its model code `ZBL12V60ATCC` uses the same "ZB" prefix as this board's `ZB16ST-C4`.
**Likely the same maker, not confirmed.**

Drive files (same manual, several copies):
- `ZBL12V60ATCC - Shenzhen Smart Li-ion.pdf` (spec pages only)
- `60A CAN li-ion (3 6V) Specification_shenzen BMS.pdf` (full 21-page manual, Rev CC-A8, 2022-04-02)

Key facts **for that product** (4S-16S, 170x60x25 mm, 60 A). These are defaults; the actual board's shipment report governs:
- Communication: isolated CAN at 500 kbit/s (firmware upgrade over CAN); non-isolated UART at 9600 baud,
  to be used only with the maker's USB-TTL PC adapter, Bluetooth or LCD module. Do not tie UART GND to an external RS485 ground.
- Default NMC thresholds: OVP 4.25 V (release 4.10 V), UVP 2.80 V (release 3.30 V), 1 s delays.
  Discharge OCP 110 A / 120 A (1 s), charge OCP 55 A, short-circuit 200-600 us.
- Temperature: discharge cutoff 65 / -20 C, charge cutoff 55 / -5 C, MOSFET 85 C. Up to 4 NTC probes.
- Passive balancing, 75 ohm (10-30 mA). Starts above 3.8 V when cell difference >50 mV; stops below 20 mV.
- Sleep after 30 s idle (<300 uA); wakes on current or communication. Deep sleep after over-discharge (<30 uA),
  which only a charger can wake.
- Reset: a pair of white wires; shorting them restarts the BMS.
- Connection order (from the manual): solder the balance leads from lowest to highest cell without plugging them in,
  then B-, P- and B+, then plug in the balance connector, then connect the charger or load. Remove in reverse order.
- Parameters can be changed with their PC software, but it is password-protected; request it from the maker.

Next step: email the maker (address in the manual) with the board markings and photos and ask for the ZB16ST-C4 spec,
the UART/CAN protocol document and the PC tool.

Other BMS documents in the drive are for the HV accumulator's Orion BMS 2, not for this board.

---

## Update: full photo of both packs (2026-11-22)

Read from the second photo (two packs side by side, each with its own BMS):
- Each BMS has two parts: the black control board (`ZB16ST-C4-V8.5` on the right pack; the left one has the same layout)
  and a separate **power board marked "2021.08.12 V4.2"**. The power board carries two columns of power MOSFETs
  (charge and discharge switches) and the DFET / RSENSE / CFET / PSENSE / ISP / ISM / B- pins that join it to the control board.
- The power-board pads are labelled B- and P/C. "P/C" suggests one shared terminal for charge and discharge (inferred).
- A black multi-wire cable runs between the two BMSs' communication connectors.
- A cell sleeve shows "INR-21700-P4…" and "NOT FOR E-CIG". This is probably Molicel INR21700-P45B
  (the HV accumulator cell) (inferred, partly obscured).
- Thick single power leads leave the top of each pack.

Implications:
- Separate control and power boards (one model code each) is how the Shenzhen Smart Li-ion manual describes its
  large-pack range. That still fits the "ZB" match but doesn't confirm it.
- **Caution:** that manual says the BMS "can only be used in a separate battery pack, can not in parallel or in series".
  Check what the black cable between the two BMSs is, and whether the packs are meant to be joined,
  before connecting them together.
- The ISM/ISP pins are likely the current-shunt sense lines (I-sense minus/plus) (inferred).
