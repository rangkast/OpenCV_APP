from robot_system.resource.definition import *
from robot_system.resource.robot_system_data import *
TAG = '[BLUETOOTH]'


class Scanner:
    def __init__(self):
        self._scanner = BleakScanner()
        self._scanner.register_detection_callback(self.detection_callback)
        self.scanning = asyncio.Event()

    def detection_callback(self, device, advertisement_data):
        # Looking for:
        ble_info = str(device).split(': ')

        if DEBUG > DEBUG_LEVEL.LV_2:
            print(TAG, ble_info)
        if DEVICE_LOOK in ble_info[1]:
            print(TAG, ble_info, ' detected')
            ROBOT_SYSTEM_DATA[SYSTEM_SETTING].set_bt_data(BT_DATA(ble_info[0], ble_info[1], NOT_SET))
            self.scanning.clear()

    async def run(self, loop):
        await self._scanner.start()
        self.scanning.set()
        end_time = loop.time() + TIMEOUT_SECONDS
        while self.scanning.is_set():
            if loop.time() > end_time:
                self.scanning.clear()
                print(TAG, '\t\tScan has timed out so we terminate')
            await asyncio.sleep(0.1)
        await self._scanner.stop()


async def characteristic_info(address):
    async with BleakClient(address) as client:
        print(TAG, 'characteristic info')
        services = await client.get_services()
        char_list = []
        for service in services:
            if DEBUG > DEBUG_LEVEL.DISABLE:
                print(TAG, service)
            if service.uuid == BASE_SERVICE_UUID:
                for characteristic in service.characteristics:
                    char_list.append({'uuid': characteristic.uuid,
                                      'desc': characteristic.description,
                                      'property': characteristic.properties})
                bt_data = copy.deepcopy(ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_bt_data())
                bt_data.list = char_list
                ROBOT_SYSTEM_DATA[SYSTEM_SETTING].set_bt_data(bt_data)


async def read_write(cmd, uuid, address):
    async with BleakClient(address) as client:
        print(TAG, 'connected')
        services = await client.get_services()
        for service in services:
            for characteristic in service.characteristics:
                if characteristic.uuid != uuid:
                    continue
                print(TAG, characteristic, ' try to write:', cmd)
                if CAL_RX_CHAR_UUID == uuid:
                    if cmd == 'W':
                        pts = ROBOT_SYSTEM_DATA[LED_INFO]
                        for i in range(len(pts)):
                            if len(pts[i].get_remake_3d()) > 0:
                                idx = pts[i].get_idx()
                                x = pts[i].get_remake_3d()[0].blob.x
                                y = pts[i].get_remake_3d()[0].blob.y
                                z = pts[i].get_remake_3d()[0].blob.z
                                u = pts[i].get_dir()[0][0]
                                v = pts[i].get_dir()[0][1]
                                w = pts[i].get_dir()[0][2]
                                str_data = ''.join([f'{cmd} ',
                                                    f'{idx} ',
                                                    f'{x} ', f'{y} ', f'{z} ',
                                                    f'{u} ', f'{v} ', f'{w}'])
                                if DEBUG > DEBUG_LEVEL.LV1:
                                    print(TAG, str_data)
                                await asyncio.sleep(0.3)
                                await client.write_gatt_char(characteristic, str_data.encode('utf-8'))
                elif CAL_TX_CHAR_UUID == uuid:
                    if 'notify' in characteristic.properties:
                        str_data = ''.join([f'{cmd} '])
                        print(TAG, str_data)
                        await client.write_gatt_char(CAL_RX_CHAR_UUID, str_data.encode('utf-8'))
                        print(TAG, 'try to activate notify.')
                        await client.start_notify(characteristic, notify_callback)

        if client.is_connected:
            await asyncio.sleep(10)
            if cmd == 'R':
                print(TAG, 'try to deactivate notify.')
                await client.stop_notify(CAL_TX_CHAR_UUID)

    print(TAG, 'disconnect')


def notify_callback(sender: int, data: bytearray):
    # print('sender:', sender, ':', data, ' len:', len(data))
    x = struct.unpack('f', data[0:4])
    y = struct.unpack('f', data[4:8])
    z = struct.unpack('f', data[8:12])
    u = struct.unpack('f', data[12:16])
    v = struct.unpack('f', data[16:20])
    w = struct.unpack('f', data[20:24])
    led_num = struct.unpack("B", data[24:25])

    str_data = ''.join([f'{led_num[0]} ',
                        f'{round(x[0], 8)} ', f'{round(y[0], 8)} ', f'{round(z[0], 8)} ',
                        f'{round(u[0], 8)} ', f'{round(v[0], 8)} ', f'{round(w[0], 8)}'])
    if DEBUG > DEBUG_LEVEL.LV1:
        print(TAG, str_data)


def bt_cal_write():
    if DEBUG > DEBUG_LEVEL.LV1:
        print(TAG, 'try to connect ', ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_bt_data().mac)
    asyncio.get_event_loop().run_until_complete(
        read_write('W', CAL_RX_CHAR_UUID, ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_bt_data().mac))
    print(TAG, 'write done')
    return


def bt_cal_read():
    asyncio.get_event_loop().run_until_complete(
        read_write('R', CAL_TX_CHAR_UUID, ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_bt_data().mac))
    print(TAG, 'read done')
    return


# ToDo
def bt_control_pwm(freq):
    return


def robot_bt_send():
    print(TAG, robot_bt_send.__name__)
    loop = asyncio.get_event_loop()
    my_scanner = Scanner()
    loop.run_until_complete(my_scanner.run(loop))

    # check handler
    if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_bt_data().mac != NOT_SET:
        asyncio.get_event_loop().run_until_complete(characteristic_info(ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_bt_data().mac))

        if DEBUG > DEBUG_LEVEL.DISABLE:
            for data in ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_bt_data().list:
                print(TAG, data)
        bt_cal_write()
        bt_cal_read()

    return SUCCESS
