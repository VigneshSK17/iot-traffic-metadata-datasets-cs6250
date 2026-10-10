from hashlib import sha256
from pathlib import Path

import polars as pl

def format_iot_malware_dataset(path: str):
    df = pl.read_csv(path)
    print(f"{path} loaded: {df.shape}")

    mapping = {
        "Timestamp": "timestamp",
        "Device_ID": "device_id",
        "Device_Type": "device_type",
        "Source_Port": "src_port",
        "Destination_Port": "dst_port",
        "Flow_Duration_ms": "flow_duration_ms",
        "Packets_Per_Flow": "total_packets",
        "Bytes_Per_Flow": "total_bytes",
        "Packet_Rate": "packet_rate", # per second
        "Packet_Size": "avg_packet_size",
        "Manufacturer": "manufacturer",
        "Protocol": "transport_protocol",
        "Source_IP": "source_ip",
        "Destination_IP": "dest_ip",
    }

    df = df.rename(mapping).with_columns(
        pl.col("timestamp").str.to_datetime().alias("timestamp"),
        pl.lit("iot_malware_dataset").alias("dataset_source"),
        pl.when(pl.col("Is_Malicious") == 1)
        .then(pl.lit(1, dtype=pl.Int64))
        .otherwise(pl.lit(0, dtype=pl.Int64)).alias("is_malicious"),
        pl.when(pl.col("Is_Malicious") == 1)
        .then(pl.col("Attack_Type").cast(pl.String))
        .otherwise(pl.lit(None, dtype=pl.String)).alias("attack_type"),
    )

    print((df["total_packets"] / df["flow_duration_ms"] / 1000)[:5])
    print(df["packet_rate"][:5])

    df.write_parquet("processed/iot_malware_dataset.parquet")


def _format_unsw_iot_device(path: str):
    device_mapping = {
        "AmazonEcho_44650d56ccd3_flows.csv": ("unsw_amazon_echo", "Smart Speaker", "Amazon"),
        "AugustDoorBell_e076d03f00ae_flows.csv": ("unsw_august_doorbell", "Doorbell", "August"),
        "AwairAirQuality_70886b100fc6_flows.csv": ("unsw_awair_air_quality", "Air Quality Sensor", "Awair"),
        "BelkinCamera_b4750eece5a9_flows.csv": ("unsw_belkin_camera", "IP Camera", "Belkin"),
        "BelkinWemoMotionSensor_ec1a59832811_flows.csv": ("unsw_belkin_wemo_motion_sensor", "Motion Sensor", "Belkin"),
        "BelkinWemoSwitch_ec1a5979f489_flows.csv": ("unsw_belkin_wemo_switch", "Smart Plug", "Belkin"),
        "BlipCareBPMeter_746a89002e25_flows.csv": ("unsw_blipcare_bp_meter", "Blood Pressure Monitor", "BlipCare"),
        "CanaryCamera_7c70bc5d5edc_flows.csv": ("unsw_canary_camera", "Smart Camera", "Canary"),
        "HPPrinter_705a0fe49bc0_flows.csv": ("unsw_hp_printer", "Printer", "HP"),
        "HelloBarbie_28c2ddffa52d_flows.csv": ("unsw_hello_barbie", "Smart Toy", "Mattel"),
        "LiFXBulb_d073d5018308_flows.csv": ("unsw_lifx_bulb", "Smart Bulb", "LIFX"),
        "NestDropCam_308cfb2fe4b2_flows.csv": ("unsw_nest_dropcam", "Smart Camera", "Google"),
        "NestProtect_18b43025bee4_flows.csv": ("unsw_nest_protect", "Smoke Detector", "Google"),
        "NetatmoWeatherStation_70ee5003b8ac_flows.csv": ("unsw_netatmo_weather_station", "Weather Station", "Netatmo"),
        "NetatmoWelcome_70ee50183443_flows.csv": ("unsw_netatmo_welcome", "Smart Camera", "Netatmo"),
        "PhilipsHue_0017882b9a25_flows.csv": ("unsw_philips_hue", "Smart Hub", "Philips"),
        "PixStarPhotoFrame_e076d033bb85_flows.csv": ("unsw_pix_star_photo_frame", "Photo Frame", "Pix-Star"),
        "RingDoorBell_884aea31669d_flows.csv": ("unsw_ring_doorbell", "Doorbell", "Ring"),
        "SamsungCamera_00166cab6b88_flows.csv": ("unsw_samsung_camera", "IP Camera", "Samsung"),
        "SamsungSmartThings_d052a800675e_flows.csv": ("unsw_samsung_smartthings", "Smart Hub", "Samsung"),
        "TPLinkCamera_f4f26D9351f1_flows.csv": ("unsw_tp_link_camera", "IP Camera", "TP-Link"),
        "TPLinkSmartPlug_50c7bf005639_flows.csv": ("unsw_tp_link_smart_plug", "Smart Plug", "TP-Link"),
        "TribySpeaker_18B79E022044_flows.csv": ("unsw_triby_speaker", "Smart Speaker", "Invoxia"),
        "WithingsBabyMonitor_0024e41118a8_flows.csv": ("unsw_withings_baby_monitor", "Smart Camera", "Withings"),
        "WithingsSleepSensor_0024e42028c6_flows.csv": ("unsw_withings_sleep_sensor", "Sleep Sensor", "Withings"),
        "WithingsSmartScale_0024e41b6f96_flows.csv": ("unsw_withings_smart_scale", "Smart Scale", "Withings"),
        "iHome_74c63b29d71d_flows.csv": ("unsw_ihome", "Smart Plug", "iHome"),
    }

    filename = Path(path).name
    if filename not in device_mapping:
        raise ValueError(f"No UNSW IoT device mapping for {filename}")
    device_id, device_type, manufacturer = device_mapping[filename]

    df = pl.read_csv(
        path,
        schema_overrides={
            "srcPort": pl.String,
            "dstPort": pl.String,
            "flowDuration": pl.Float64,
        },
    )
    print(f"{path} loaded: {df.shape}")

    df = df.with_columns(
        pl.lit(device_id).alias("device_id"),
        pl.lit(device_type).alias("device_type"),
        pl.lit(manufacturer).alias("manufacturer"),
        pl.lit(0, dtype=pl.Int64).alias("is_malicious"),
        pl.lit(None, dtype=pl.String).alias("attack_type"),
        pl.col("srcIp").cast(pl.String).alias("source_ip"),
        pl.col("dstIp").cast(pl.String).alias("dest_ip"),
    )
    df = (
        df.lazy()
        .rename({
            "time": "timestamp",
            "srcPort": "src_port",
            "dstPort": "dst_port",
        })
        .with_columns(
            pl.col("src_port").cast(pl.Int64, strict=False),
            pl.col("dst_port").cast(pl.Int64, strict=False),
            (
                pl.col("flowDuration") * 1000
            ).alias("flow_duration_ms"),

            (
                pl.col("srcNumPackets") +
                pl.col("dstNumPackets")
            ).alias("total_packets"),

            (
                pl.col("srcPayloadSize") +
                pl.col("dstPayloadSize")
            ).alias("payload_bytes"),

            pl.lit("unsw").alias("dataset_source"),
        )
        .with_columns(
            pl.col("timestamp")
            .str.to_datetime(strict=False, time_zone="UTC"),
            pl.col("payload_bytes").alias("total_bytes"),
            pl.when(pl.col("flow_duration_ms") > 0)
            .then(pl.col("total_packets") * 1000 / pl.col("flow_duration_ms"))
            .otherwise(None)
            .alias("packet_rate"),
            pl.when(pl.col("total_packets") > 0)
            .then(pl.col("payload_bytes") / pl.col("total_packets"))
            .otherwise(None)
            .alias("avg_packet_size"),
        )
    )

    # protocol value
    df = df.with_columns(
        pl.when(pl.col("ipProto") == 6)
        .then(pl.lit("TCP"))
        .when(pl.col("ipProto") == 17)
        .then(pl.lit("UDP"))
        .when(pl.col("ipProto") == 1)
        .then(pl.lit("ICMP"))
        .otherwise(pl.lit("OTHER"))
        .alias("transport_protocol")
    )

    return df.collect()


def format_unsw_iot_dataset(path: str = "UNSW_IoT/flows"):
    files = sorted(Path(path).glob("*.csv"))

    df = pl.concat(
        [_format_unsw_iot_device(str(filepath)) for filepath in files],
        how="diagonal_relaxed",
    )
    output_path = Path("processed/unsw_iot_dataset.parquet")

    df.write_parquet(output_path)


def _format_cic_iot_device(path: str):
    device_mapping = {
        "amazonplug": ("Smart Plug", "Amazon"),
        "amcrestcam": ("IP Camera", "Amcrest"),
        "arlobasestationcam": ("Smart Hub", "Arlo"),
        "arloqcam": ("Smart Camera", "Arlo"),
        "atomicoffeemaker": ("Coffee Maker", "Atomi"),
        "boruncam": ("IP Camera", "Borun"),
        "dlinkcam": ("IP Camera", "D-Link"),
        "echodot1": ("Smart Speaker", "Amazon"),
        "echodot2": ("Smart Speaker", "Amazon"),
        "echodot3": ("Smart Speaker", "Amazon"),
        "echospot": ("Smart Speaker", "Amazon"),
        "echostudio": ("Smart Speaker", "Amazon"),
        "eufyhomebase": ("Smart Hub", "Eufy"),
        "globelamp": ("Smart Bulb", "Globe Electric"),
        "heimvisioncam": ("IP Camera", "HeimVision"),
        "heimvisionlamp": ("Smart Bulb", "HeimVision"),
        "heimvision": ("Smart Bulb", "HeimVision"),
        "homeeyecam": ("IP Camera", "HomeEye"),
        "luohecam": ("IP Camera", "Luohe"),
        "nestcam": ("Smart Camera", "Google"),
        "nestmini": ("Smart Speaker", "Google"),
        "netatmocam": ("Smart Camera", "Netatmo"),
        "philipshue": ("Smart Bulb", "Philips"),
        "roomba": ("Robot Vacuum", "iRobot"),
        "simcam": ("Smart Camera", "SimCam"),
        "smartboard": ("Smart Board", "Unknown"),
        "sonosone": ("Smart Speaker", "Sonos"),
        "teckin1": ("Smart Plug", "Teckin"),
        "teckin2": ("Smart Plug", "Teckin"),
        "yutron1": ("Smart Plug", "Yutron"),
        "yutron2": ("Smart Plug", "Yutron"),
    }
    filename = Path(path).stem.lower()
    device = next(
        name for name in sorted(device_mapping, key=len, reverse=True)
        if filename.startswith(name)
    )
    if device == "heimvision":
        device = "heimvisionlamp"
    device_type, manufacturer = device_mapping[device]

    df = pl.read_csv(path, infer_schema_length=0)
    print(f"{path} loaded: {df.shape}")
    df = df.rename({name: name.strip() for name in df.columns}).drop("Unnamed: 0")
    df = df.with_columns(pl.all().cast(pl.Float64)).with_columns(
        pl.col("source_port").cast(pl.Int64).alias("src_port"),
        pl.col("dest_port").cast(pl.Int64).alias("dst_port"),
        pl.col("protocol").cast(pl.Int64),
        pl.col("ip_dst_new").cast(pl.Int64),
    )

    group_keys = ["ip_dst_new", "src_port", "dst_port", "protocol"]
    df = (
        df.lazy()
        .sort("epoch_timestamp")
        .with_columns(
            ((pl.col("epoch_timestamp").cum_count().over(group_keys) - 1) // 20)
            .alias("window_id")
        )
        .group_by(group_keys + ["window_id"], maintain_order=True)
        .agg(
            pl.exclude(group_keys + ["window_id"]).last(),
            pl.col("epoch_timestamp").min().alias("window_start"),
            ((pl.col("epoch_timestamp").max() - pl.col("epoch_timestamp").min()) * 1000)
            .alias("flow_duration_ms"),
            pl.len().cast(pl.Int64).alias("total_packets"),
            pl.col("total_length").sum().cast(pl.Int64).alias("total_bytes"),
            pl.col("total_length").mean().alias("avg_packet_size"),
        )
        .with_columns(
            pl.from_epoch((pl.col("window_start") * 1_000_000).round().cast(pl.Int64), time_unit="us")
            .dt.replace_time_zone("UTC").alias("timestamp"),
            pl.lit(f"cic_{device}").alias("device_id"),
            pl.lit(device_type).alias("device_type"),
            pl.lit(manufacturer).alias("manufacturer"),
            pl.lit("cic_iot_dataset2022").alias("dataset_source"),
            pl.lit(0, dtype=pl.Int64).alias("is_malicious"),
            pl.lit(None, dtype=pl.String).alias("attack_type"),
            pl.lit(None, dtype=pl.String).alias("source_ip"),
            pl.concat_str(
                [(pl.col("ip_dst_new") // (256 ** shift) % 256).cast(pl.String)
                 for shift in range(3, -1, -1)],
                separator=".",
            ).alias("dest_ip"),
            pl.lit("20_packet_window").alias("aggregation_type"),
            pl.lit(str(path)).alias("source_file"),
            pl.when(pl.col("protocol") == 6).then(pl.lit("TCP"))
            .when(pl.col("protocol") == 17).then(pl.lit("UDP"))
            .when(pl.col("protocol") == 1).then(pl.lit("ICMP"))
            .otherwise(pl.lit("OTHER")).alias("transport_protocol"),
            pl.when(pl.col("flow_duration_ms") > 0)
            .then(pl.col("total_packets") * 1000 / pl.col("flow_duration_ms"))
            .otherwise(None).alias("packet_rate"),
        )
        .drop("window_start")
    )
    return df.collect()


def format_cic_iot_dataset(path: str = "cic_iot_dataset2022/CSV files/28 devices"):
    files = sorted(
        filepath for filepath in Path(path).rglob("*.csv")
        if not filepath.name.startswith("._")
    )
    fingerprints = set()
    frames = []
    for filepath in files:
        fingerprint = (filepath.name, sha256(filepath.read_bytes()).digest())
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        frames.append(_format_cic_iot_device(str(filepath)))

    df = pl.concat(frames, how="diagonal_relaxed")
    print(f"CIC IoT dataset combined: {df.shape}")
    df.write_parquet("processed/cic_iot_dataset2022.parquet")
    return df


if __name__ == "__main__":
    format_iot_malware_dataset("iot-malware-dataset/iot_malware_dataset.csv")
    format_unsw_iot_dataset("UNSW_IoT/flows")
    format_cic_iot_dataset()
