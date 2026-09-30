#import <AppKit/AppKit.h>
#import <CoreAudio/CoreAudio.h>
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#import <Foundation/Foundation.h>
#import <signal.h>
#import <unistd.h>

static volatile sig_atomic_t shouldStop = 0;
static AudioObjectID tapID = kAudioObjectUnknown;
static AudioObjectID aggregateID = kAudioObjectUnknown;
static AudioDeviceIOProcID ioProcID = NULL;

static void stopSignal(int signalNumber) {
    (void)signalNumber;
    shouldStop = 1;
}

static BOOL readValue(AudioObjectID objectID, AudioObjectPropertySelector selector,
                      void *value, UInt32 size, const void *qualifier, UInt32 qualifierSize) {
    AudioObjectPropertyAddress address = {
        selector, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain
    };
    return AudioObjectGetPropertyData(objectID, &address, qualifierSize, qualifier,
                                      &size, value) == noErr;
}

static NSArray<NSDictionary *> *audioProcesses(void) {
    AudioObjectPropertyAddress address = {
        kAudioHardwarePropertyProcessObjectList,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &address, 0, NULL, &size) != noErr) {
        return @[];
    }
    NSUInteger count = size / sizeof(AudioObjectID);
    AudioObjectID *objects = calloc(count, sizeof(AudioObjectID));
    if (!objects) return @[];
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &address, 0, NULL, &size, objects) != noErr) {
        free(objects);
        return @[];
    }

    NSMutableArray<NSDictionary *> *result = [NSMutableArray array];
    pid_t selfPID = getpid();
    for (NSUInteger index = 0; index < count; index++) {
        AudioObjectID objectID = objects[index];
        pid_t pid = -1;
        UInt32 isOutputting = 0;
        if (!readValue(objectID, kAudioProcessPropertyPID, &pid, sizeof(pid), NULL, 0) ||
            pid == selfPID ||
            !readValue(objectID, kAudioProcessPropertyIsRunningOutput, &isOutputting,
                       sizeof(isOutputting), NULL, 0) || !isOutputting) {
            continue;
        }

        NSString *name = nil;
        NSRunningApplication *app = [NSRunningApplication runningApplicationWithProcessIdentifier:pid];
        name = app.localizedName;
        if (!name.length) {
            CFStringRef bundleID = NULL;
            if (readValue(objectID, kAudioProcessPropertyBundleID, &bundleID,
                          sizeof(bundleID), NULL, 0) && bundleID) {
                name = (__bridge_transfer NSString *)bundleID;
            }
        }
        if (!name.length) name = [NSString stringWithFormat:@"Processus %d", pid];

        [result addObject:@{
            @"object_id": @(objectID),
            @"pid": @(pid),
            @"name": name,
            @"bundle_id": app.bundleIdentifier ?: @""
        }];
    }
    free(objects);
    [result sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        return [a[@"name"] localizedCaseInsensitiveCompare:b[@"name"]];
    }];
    return result;
}

static BOOL writeAll(const void *bytes, size_t length) {
    const uint8_t *cursor = bytes;
    while (length > 0) {
        ssize_t written = write(STDOUT_FILENO, cursor, length);
        if (written <= 0) return NO;
        cursor += written;
        length -= (size_t)written;
    }
    return YES;
}

static OSStatus setupTap(AudioObjectID processObjectID, AudioStreamBasicDescription *formatOut) {
    CATapDescription *tapDescription = [[CATapDescription alloc]
        initMonoMixdownOfProcesses:@[@(processObjectID)]];
    tapDescription.name = @"GEKI live capture";
    tapDescription.UUID = [NSUUID UUID];
    tapDescription.privateTap = YES;
    tapDescription.muteBehavior = CATapUnmuted;

    OSStatus status = AudioHardwareCreateProcessTap(tapDescription, &tapID);
    if (status != noErr) return status;

    AudioObjectPropertyAddress formatAddress = {
        kAudioTapPropertyFormat,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    UInt32 formatSize = sizeof(*formatOut);
    status = AudioObjectGetPropertyData(tapID, &formatAddress, 0, NULL, &formatSize, formatOut);
    if (status != noErr) return status;
    if (formatOut->mFormatID != kAudioFormatLinearPCM ||
        !(formatOut->mFormatFlags & kAudioFormatFlagIsFloat) ||
        formatOut->mBitsPerChannel != 32 || formatOut->mChannelsPerFrame != 1) {
        return kAudioDeviceUnsupportedFormatError;
    }

    NSDictionary *tapEntry = @{
        @kAudioSubTapUIDKey: tapDescription.UUID.UUIDString,
        @kAudioSubTapDriftCompensationKey: @YES
    };

    AudioObjectID outputDevice = kAudioObjectUnknown;
    AudioObjectPropertyAddress outputAddress = {
        kAudioHardwarePropertyDefaultSystemOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    UInt32 outputSize = sizeof(outputDevice);
    status = AudioObjectGetPropertyData(kAudioObjectSystemObject, &outputAddress, 0, NULL,
                                        &outputSize, &outputDevice);
    if (status != noErr) return status;

    AudioObjectPropertyAddress uidAddress = {
        kAudioDevicePropertyDeviceUID,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    CFStringRef outputUIDRef = NULL;
    UInt32 uidSize = sizeof(outputUIDRef);
    status = AudioObjectGetPropertyData(outputDevice, &uidAddress, 0, NULL, &uidSize, &outputUIDRef);
    if (status != noErr || !outputUIDRef) return status != noErr ? status : kAudioHardwareBadDeviceError;
    NSString *outputUID = CFBridgingRelease(outputUIDRef);

    NSDictionary *aggregateDescription = @{
        @kAudioAggregateDeviceNameKey: @"GEKI Live Tap",
        @kAudioAggregateDeviceUIDKey: [NSUUID UUID].UUIDString,
        @kAudioAggregateDeviceMainSubDeviceKey: outputUID,
        @kAudioAggregateDeviceIsPrivateKey: @YES,
        @kAudioAggregateDeviceIsStackedKey: @NO,
        @kAudioAggregateDeviceTapAutoStartKey: @YES,
        @kAudioAggregateDeviceSubDeviceListKey: @[@{@kAudioSubDeviceUIDKey: outputUID}],
        @kAudioAggregateDeviceTapListKey: @[tapEntry]
    };
    return AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)aggregateDescription, &aggregateID);
}

static int streamProcess(AudioObjectID processObjectID) {
    AudioStreamBasicDescription format = {0};
    OSStatus status = setupTap(processObjectID, &format);
    if (status != noErr) {
        fprintf(stderr, "GEKI tap setup failed (Core Audio status %d).\n", (int)status);
        return 2;
    }

    uint32_t sampleRate = (uint32_t)llround(format.mSampleRate);
    if (!writeAll("GEKI", 4) || !writeAll(&sampleRate, sizeof(sampleRate))) return 3;

    dispatch_queue_t queue = dispatch_queue_create("geki.audio-tap", DISPATCH_QUEUE_SERIAL);
    status = AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggregateID, queue,
        ^(const AudioTimeStamp *now, const AudioBufferList *input,
          const AudioTimeStamp *inputTime, AudioBufferList *output,
          const AudioTimeStamp *outputTime) {
            (void)now; (void)inputTime; (void)output; (void)outputTime;
            if (shouldStop || !input || input->mNumberBuffers == 0) return;
            const AudioBuffer *buffer = &input->mBuffers[0];
            if (buffer->mData && buffer->mDataByteSize > 0) {
                if (!writeAll(buffer->mData, buffer->mDataByteSize)) shouldStop = 1;
            }
        });
    if (status == noErr) status = AudioDeviceStart(aggregateID, ioProcID);
    if (status != noErr) {
        fprintf(stderr, "GEKI audio stream failed to start (Core Audio status %d).\n", (int)status);
        return 4;
    }

    signal(SIGINT, stopSignal);
    signal(SIGTERM, stopSignal);
    while (!shouldStop) usleep(100000);

    AudioDeviceStop(aggregateID, ioProcID);
    AudioDeviceDestroyIOProcID(aggregateID, ioProcID);
    ioProcID = NULL;
    AudioHardwareDestroyAggregateDevice(aggregateID);
    aggregateID = kAudioObjectUnknown;
    AudioHardwareDestroyProcessTap(tapID);
    tapID = kAudioObjectUnknown;
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc == 2 && strcmp(argv[1], "--list") == 0) {
            NSData *data = [NSJSONSerialization dataWithJSONObject:audioProcesses()
                                                           options:NSJSONWritingFragmentsAllowed
                                                             error:nil];
            fwrite(data.bytes, 1, data.length, stdout);
            fputc('\n', stdout);
            return 0;
        }
        if (argc == 3 && strcmp(argv[1], "--stream") == 0) {
            return streamProcess((AudioObjectID)strtoul(argv[2], NULL, 10));
        }
        fprintf(stderr, "Usage: GekiAudioTap --list | --stream PROCESS_OBJECT_ID\n");
        return 64;
    }
}
