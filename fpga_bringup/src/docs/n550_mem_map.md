|Start       | End         | size | Slave      | bus width | SO  | Cacheable  | Executable | U-mode | LSU access | IFU access | Fbus Access | VLSU   | lr/sc | amo   | Comment                     |
| ---        | ---         | ---  | ---        | ---       | --- | ---        | ---        | ---    | ---        | ---        | ---         | ---    | ---   | ---   |---                          |
|0x0000_0000 | 0x0000_0FFF | 4K   | Debug      | 64        | 1   | 0          | 1          | 0      | Yes        | Yes        | No          | No     | No    | No    |                             |
|0x0200_0000 | 0x0200_FFFF | 64K  | CLINT      | 64        | 1   | 0          | 0          | 0      | Yes        | No         | No          | No     | No    | No    |                             |
|0x0300_0000 | 0x0300_FFFF | 64K  | APLIC      | 64        | 1   | 0          | 0          | 0      | Yes        | No         | Yes         | No     | No    | No    |                             |
|0x0400_0000 | 0x0401_FFFF | 128K | ITCM       | 64        | 0   | 0          | 1          | 1      | Yes        | Yes        | Yes         | Yes    | Yes   | Yes   |                             |
|0x0402_0000 | 0x0403_FFFF | 128K | DTCM       | 64        | 0   | 0          | 0          | 1      | Yes        | No         | Yes         | Yes    | Yes   | Yes   |                             |
|0x1000_0000 | 0x1FFF_FFFF | 256M | CLP        | 1024      | 0   | 0          | 0          | 1      | No         | No         | No          | Yes    | No    | No    | NOC redirect to 0xF000_0000 |
|0x2000_0000 | 0x3FFF_FFFF | 512M | Peripheral | 64        | 1   | 1(IC)      | 1          | 1      | Yes        | Yes        | No          | No     | Yes   | No    | mmio                        |
|0x4000_0000 | 0x5FFF_FFFF | 512M | system     | 64        | 0   | 1(IC)      | 1          | 1      | Yes        | Yes        | No          | Yes    | Yes   | No    |                             |
|0x8000_0000 | 0xFFFF_FFFF | 2G   | Memory     | 128       | 0   | 1(IC/DC)   | 1          | 1      | Yes        | Yes        | No          | Yes    | by DC | by DC |                             |
|0x8000_0000 | 0xFFFF_FFFF | 2G   | AMU Shared | 512       | 0   | 0          | 0          | 1      | Yes        | Yes        | No          | Yes    | No    | No    |                             |
