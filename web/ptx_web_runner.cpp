#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.hpp"
#include "parser/parser.hpp"
#include "vm.hpp"

namespace {

struct CapturedStreams {
    std::ostringstream stdoutCapture;
    std::ostringstream stderrCapture;
    std::streambuf* originalStdout = nullptr;
    std::streambuf* originalStderr = nullptr;

    CapturedStreams() {
        originalStdout = std::cout.rdbuf(stdoutCapture.rdbuf());
        originalStderr = std::cerr.rdbuf(stderrCapture.rdbuf());
    }

    ~CapturedStreams() {
        if (originalStdout != nullptr) {
            std::cout.rdbuf(originalStdout);
        }
        if (originalStderr != nullptr) {
            std::cerr.rdbuf(originalStderr);
        }
    }

    std::string combined() const {
        std::string output = stdoutCapture.str();
        output += stderrCapture.str();
        return output;
    }
};

struct ScalarArgument {
    std::string name;
    std::string value;
};

struct PointerArgument {
    std::string name;
    std::string bufferType;
    std::size_t elementCount = 0;
    std::vector<std::string> rawValues;
};

struct PreparedScalar {
    std::string name;
    std::string type;
    std::string value;
};

struct PreparedPointerBuffer {
    std::string name;
    std::string bufferType;
    std::size_t elementCount = 0;
    std::size_t byteSize = 0;
    CUdeviceptr deviceAddress = 0;
    std::vector<uint8_t> beforeBytes;
    std::vector<uint8_t> afterBytes;
};

struct MemoryWatch {
    uint64_t address = 0;
    std::vector<uint8_t> bytes;
};

std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        parts.push_back(item);
    }
    return parts;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream escaped;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\b':
                escaped << "\\b";
                break;
            case '\f':
                escaped << "\\f";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    escaped << "\\u"
                            << std::hex
                            << std::setw(4)
                            << std::setfill('0')
                            << static_cast<int>(ch)
                            << std::dec
                            << std::setfill(' ');
                } else {
                    escaped << static_cast<char>(ch);
                }
                break;
        }
    }
    return escaped.str();
}

std::string quoteJson(const std::string& value) {
    return "\"" + jsonEscape(value) + "\"";
}

std::string joinJsonStringArray(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << quoteJson(values[i]);
    }
    out << "]";
    return out.str();
}

std::string formatHexAddress(uint64_t address) {
    std::ostringstream out;
    out << "0x" << std::hex << address << std::dec;
    return out.str();
}

std::string formatHexPreview(const std::vector<uint8_t>& bytes, std::size_t limit) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    const std::size_t previewSize = std::min(limit, bytes.size());
    for (std::size_t i = 0; i < previewSize; ++i) {
        if (i != 0) {
            out << " ";
        }
        out << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    }
    if (bytes.size() > previewSize) {
        out << " ...";
    }
    return out.str();
}

std::array<unsigned int, 3> parseTriplet(const std::string& rawValue, const std::string& flagName) {
    const auto parts = split(rawValue, ',');
    if (parts.size() != 3) {
        throw std::runtime_error(flagName + " expects three comma-separated integers");
    }

    std::array<unsigned int, 3> dims{};
    for (std::size_t i = 0; i < 3; ++i) {
        const std::string part = trim(parts[i]);
        unsigned long parsed = std::stoul(part, nullptr, 10);
        if (parsed == 0 || parsed > std::numeric_limits<unsigned int>::max()) {
            throw std::runtime_error(flagName + " values must be between 1 and 4294967295");
        }
        dims[i] = static_cast<unsigned int>(parsed);
    }
    return dims;
}

ScalarArgument parseScalarArgument(const std::string& rawValue) {
    const auto separator = rawValue.find('=');
    if (separator == std::string::npos || separator == 0 || separator == rawValue.size() - 1) {
        throw std::runtime_error("--scalar expects the form name=value");
    }

    ScalarArgument argument;
    argument.name = trim(rawValue.substr(0, separator));
    argument.value = trim(rawValue.substr(separator + 1));
    if (argument.name.empty() || argument.value.empty()) {
        throw std::runtime_error("--scalar expects a non-empty name and value");
    }
    return argument;
}

PointerArgument parsePointerArgument(const std::string& rawValue) {
    const auto nameSeparator = rawValue.find('=');
    if (nameSeparator == std::string::npos || nameSeparator == 0 || nameSeparator == rawValue.size() - 1) {
        throw std::runtime_error("--pointer expects the form name=type@count:value1,value2");
    }

    PointerArgument argument;
    argument.name = trim(rawValue.substr(0, nameSeparator));

    const std::string definition = rawValue.substr(nameSeparator + 1);
    const auto countSeparator = definition.find('@');
    if (countSeparator == std::string::npos || countSeparator == 0 || countSeparator == definition.size() - 1) {
        throw std::runtime_error("--pointer expects type@count before the optional values section");
    }

    const auto valuesSeparator = definition.find(':', countSeparator + 1);
    const std::string typePart = trim(definition.substr(0, countSeparator));
    const std::string countPart = trim(
        valuesSeparator == std::string::npos
            ? definition.substr(countSeparator + 1)
            : definition.substr(countSeparator + 1, valuesSeparator - countSeparator - 1));

    argument.bufferType = typePart;
    argument.elementCount = std::stoull(countPart, nullptr, 10);
    if (argument.bufferType.empty() || argument.elementCount == 0) {
        throw std::runtime_error("--pointer expects a supported buffer type and a positive element count");
    }

    if (valuesSeparator != std::string::npos && valuesSeparator + 1 < definition.size()) {
        const std::string valuesPart = trim(definition.substr(valuesSeparator + 1));
        if (!valuesPart.empty()) {
            argument.rawValues = split(valuesPart, ',');
            for (std::string& item : argument.rawValues) {
                item = trim(item);
            }
        }
    }

    return argument;
}

std::size_t bufferElementSize(const std::string& bufferType) {
    if (bufferType == "int32" || bufferType == "uint32" || bufferType == "float32") {
        return 4;
    }
    if (bufferType == "int64" || bufferType == "uint64" || bufferType == "float64") {
        return 8;
    }
    if (bufferType == "bytes") {
        return 1;
    }
    throw std::runtime_error("Unsupported pointer buffer type: " + bufferType);
}

void writeBufferValue(std::vector<uint8_t>& bytes, std::size_t offset, const std::string& bufferType, const std::string& rawValue) {
    if (bufferType == "int32") {
        const int32_t value = static_cast<int32_t>(std::stol(rawValue, nullptr, 0));
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
        return;
    }
    if (bufferType == "uint32") {
        const uint32_t value = static_cast<uint32_t>(std::stoul(rawValue, nullptr, 0));
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
        return;
    }
    if (bufferType == "float32") {
        const float value = std::stof(rawValue);
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
        return;
    }
    if (bufferType == "int64") {
        const int64_t value = std::stoll(rawValue, nullptr, 0);
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
        return;
    }
    if (bufferType == "uint64") {
        const uint64_t value = std::stoull(rawValue, nullptr, 0);
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
        return;
    }
    if (bufferType == "float64") {
        const double value = std::stod(rawValue);
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
        return;
    }
    if (bufferType == "bytes") {
        const unsigned long value = std::stoul(rawValue, nullptr, 0);
        if (value > 255UL) {
            throw std::runtime_error("Byte buffer values must be between 0 and 255");
        }
        bytes[offset] = static_cast<uint8_t>(value);
        return;
    }

    throw std::runtime_error("Unsupported pointer buffer type: " + bufferType);
}

std::vector<uint8_t> preparePointerBytes(const PointerArgument& argument) {
    const std::size_t elementSize = bufferElementSize(argument.bufferType);
    std::vector<uint8_t> bytes(argument.elementCount * elementSize, 0);

    if (argument.rawValues.size() > argument.elementCount) {
        throw std::runtime_error(
            "Pointer parameter '" + argument.name + "' received more initial values than its declared element count");
    }

    for (std::size_t index = 0; index < argument.rawValues.size(); ++index) {
        writeBufferValue(bytes, index * elementSize, argument.bufferType, argument.rawValues[index]);
    }

    return bytes;
}

void packScalarBits(const PTXParameter& parameter, const std::string& rawValue, uint64_t& packedBits, std::string& displayValue) {
    packedBits = 0;

    if (parameter.type == ".u8" || parameter.type == ".b8") {
        const uint8_t value = static_cast<uint8_t>(std::stoul(rawValue, nullptr, 0));
        std::memcpy(&packedBits, &value, sizeof(value));
        displayValue = std::to_string(static_cast<unsigned int>(value));
        return;
    }
    if (parameter.type == ".s8") {
        const int8_t value = static_cast<int8_t>(std::stoi(rawValue, nullptr, 0));
        std::memcpy(&packedBits, &value, sizeof(value));
        displayValue = std::to_string(static_cast<int>(value));
        return;
    }
    if (parameter.type == ".u16" || parameter.type == ".b16") {
        const uint16_t value = static_cast<uint16_t>(std::stoul(rawValue, nullptr, 0));
        std::memcpy(&packedBits, &value, sizeof(value));
        displayValue = std::to_string(value);
        return;
    }
    if (parameter.type == ".s16") {
        const int16_t value = static_cast<int16_t>(std::stoi(rawValue, nullptr, 0));
        std::memcpy(&packedBits, &value, sizeof(value));
        displayValue = std::to_string(value);
        return;
    }
    if (parameter.type == ".u32" || parameter.type == ".b32") {
        const uint32_t value = static_cast<uint32_t>(std::stoul(rawValue, nullptr, 0));
        std::memcpy(&packedBits, &value, sizeof(value));
        displayValue = std::to_string(value);
        return;
    }
    if (parameter.type == ".s32") {
        const int32_t value = static_cast<int32_t>(std::stol(rawValue, nullptr, 0));
        std::memcpy(&packedBits, &value, sizeof(value));
        displayValue = std::to_string(value);
        return;
    }
    if (parameter.type == ".u64" || parameter.type == ".b64") {
        const uint64_t value = std::stoull(rawValue, nullptr, 0);
        std::memcpy(&packedBits, &value, sizeof(value));
        displayValue = std::to_string(value);
        return;
    }
    if (parameter.type == ".s64") {
        const int64_t value = std::stoll(rawValue, nullptr, 0);
        std::memcpy(&packedBits, &value, sizeof(value));
        displayValue = std::to_string(value);
        return;
    }
    if (parameter.type == ".f32") {
        const float value = std::stof(rawValue);
        std::memcpy(&packedBits, &value, sizeof(value));
        std::ostringstream out;
        out << std::setprecision(7) << value;
        displayValue = out.str();
        return;
    }
    if (parameter.type == ".f64") {
        const double value = std::stod(rawValue);
        std::memcpy(&packedBits, &value, sizeof(value));
        std::ostringstream out;
        out << std::setprecision(15) << value;
        displayValue = out.str();
        return;
    }

    throw std::runtime_error("Unsupported scalar PTX type: " + parameter.type);
}

std::vector<std::string> decodePreviewValues(const std::vector<uint8_t>& bytes, const std::string& bufferType, std::size_t previewCount) {
    std::vector<std::string> values;
    const std::size_t elementSize = bufferElementSize(bufferType);
    const std::size_t totalElements = bytes.size() / elementSize;
    const std::size_t count = std::min(previewCount, totalElements);
    values.reserve(count);

    for (std::size_t index = 0; index < count; ++index) {
        const uint8_t* data = bytes.data() + index * elementSize;
        std::ostringstream out;
        if (bufferType == "int32") {
            int32_t value = 0;
            std::memcpy(&value, data, sizeof(value));
            out << value;
        } else if (bufferType == "uint32") {
            uint32_t value = 0;
            std::memcpy(&value, data, sizeof(value));
            out << value;
        } else if (bufferType == "float32") {
            float value = 0.0f;
            std::memcpy(&value, data, sizeof(value));
            out << std::setprecision(7) << value;
        } else if (bufferType == "int64") {
            int64_t value = 0;
            std::memcpy(&value, data, sizeof(value));
            out << value;
        } else if (bufferType == "uint64") {
            uint64_t value = 0;
            std::memcpy(&value, data, sizeof(value));
            out << value;
        } else if (bufferType == "float64") {
            double value = 0.0;
            std::memcpy(&value, data, sizeof(value));
            out << std::setprecision(15) << value;
        } else if (bufferType == "bytes") {
            out << static_cast<unsigned int>(*data);
        } else {
            throw std::runtime_error("Unsupported pointer buffer type: " + bufferType);
        }
        values.push_back(out.str());
    }

    return values;
}

std::vector<std::string> decodeInt32Preview(const std::vector<uint8_t>& bytes, std::size_t previewCount) {
    std::vector<std::string> values;
    if (bytes.size() < sizeof(int32_t)) {
        return values;
    }

    const std::size_t totalElements = bytes.size() / sizeof(int32_t);
    const std::size_t count = std::min(previewCount, totalElements);
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        int32_t value = 0;
        std::memcpy(&value, bytes.data() + index * sizeof(int32_t), sizeof(value));
        values.push_back(std::to_string(value));
    }
    return values;
}

std::string joinJsonValueArray(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

const PTXFunction* findKernel(const PTXProgram& program, const std::string& kernelName, std::size_t& entryIndex) {
    for (std::size_t idx = 0; idx < program.entryPoints.size(); ++idx) {
        const std::size_t functionIndex = program.entryPoints[idx];
        if (functionIndex >= program.functions.size()) {
            continue;
        }
        const PTXFunction& function = program.functions[functionIndex];
        if (function.isEntry && function.name == kernelName) {
            entryIndex = functionIndex;
            return &function;
        }
    }
    return nullptr;
}

std::string availableKernelNames(const PTXProgram& program) {
    std::vector<std::string> names;
    for (std::size_t functionIndex : program.entryPoints) {
        if (functionIndex < program.functions.size()) {
            names.push_back(program.functions[functionIndex].name);
        }
    }

    std::ostringstream out;
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << names[i];
    }
    return out.str();
}

std::vector<MemoryWatch> collectMemoryWatches(PTXVM& vm) {
    const std::array<uint64_t, 2> addresses = {0x10000ULL, 0x20000ULL};
    const std::array<std::size_t, 2> sizes = {64U, 32U};

    std::vector<MemoryWatch> watches;
    for (std::size_t i = 0; i < addresses.size(); ++i) {
        MemoryWatch watch;
        watch.address = addresses[i];
        watch.bytes.resize(sizes[i], 0);
        if (vm.copyMemoryDtoH(watch.bytes.data(), watch.address, watch.bytes.size())) {
            watches.push_back(std::move(watch));
        }
    }
    return watches;
}

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  ptx_web_runner inspect <ptx_file>\n"
        << "  ptx_web_runner run <ptx_file> --kernel <name> [--grid x,y,z] [--block x,y,z]\n"
        << "                         [--scalar name=value] [--pointer name=type@count:value1,value2]\n";
}

int emitInspectJson(const std::filesystem::path& filePath) {
    PTXParser parser;
    if (!parser.parseFile(filePath.string())) {
        std::cout << "{"
                  << "\"ok\":false,"
                  << "\"error\":" << quoteJson(parser.getErrorMessage())
                  << "}";
        return 1;
    }

    const PTXProgram& program = parser.getProgram();

    std::ostringstream out;
    out << "{";
    out << "\"ok\":true,";
    out << "\"program\":{";
    out << "\"filename\":" << quoteJson(filePath.filename().string()) << ",";
    out << "\"path\":" << quoteJson(filePath.string()) << ",";
    out << "\"version\":" << quoteJson(program.metadata.version) << ",";
    out << "\"target\":" << quoteJson(program.metadata.target) << ",";
    out << "\"address_size\":" << program.metadata.addressSize;
    out << "},";

    out << "\"kernels\":[";
    bool firstKernel = true;
    for (std::size_t functionIndex : program.entryPoints) {
        if (functionIndex >= program.functions.size()) {
            continue;
        }
        const PTXFunction& function = program.functions[functionIndex];
        if (!function.isEntry) {
            continue;
        }
        if (!firstKernel) {
            out << ",";
        }
        firstKernel = false;
        out << "{";
        out << "\"name\":" << quoteJson(function.name) << ",";
        out << "\"parameter_count\":" << function.parameters.size() << ",";
        out << "\"parameters\":[";
        for (std::size_t paramIndex = 0; paramIndex < function.parameters.size(); ++paramIndex) {
            if (paramIndex != 0) {
                out << ",";
            }
            const PTXParameter& parameter = function.parameters[paramIndex];
            out << "{";
            out << "\"name\":" << quoteJson(parameter.name) << ",";
            out << "\"type\":" << quoteJson(parameter.type) << ",";
            out << "\"offset\":" << parameter.offset << ",";
            out << "\"size\":" << parameter.size << ",";
            out << "\"is_pointer\":" << (parameter.isPointer ? "true" : "false");
            out << "}";
        }
        out << "]";
        out << "}";
    }
    out << "],";
    out << "\"warnings\":" << joinJsonStringArray(program.warnings) << ",";
    out << "\"errors\":" << joinJsonStringArray(program.errors);
    out << "}";

    std::cout << out.str();
    return 0;
}

int emitRunJson(
    const std::filesystem::path& filePath,
    const std::string& requestedKernel,
    const std::array<unsigned int, 3>& gridDims,
    const std::array<unsigned int, 3>& blockDims,
    const std::vector<ScalarArgument>& scalarArgs,
    const std::vector<PointerArgument>& pointerArgs) {
    Logger::setColorOutput(false);
    Logger::setShowTimestamp(false);
    Logger::setLogLevel(LogLevel::ERROR);

    CapturedStreams capture;

    try {
        PTXParser parser;
        if (!parser.parseFile(filePath.string())) {
            throw std::runtime_error("Failed to parse PTX: " + parser.getErrorMessage());
        }

        const PTXProgram& parsedProgram = parser.getProgram();
        if (parsedProgram.entryPoints.empty()) {
            throw std::runtime_error("The PTX file does not contain any .entry kernels");
        }

        std::string kernelName = requestedKernel;
        if (kernelName.empty()) {
            if (parsedProgram.entryPoints.size() != 1) {
                throw std::runtime_error(
                    "Multiple entry kernels found. Please choose one: " + availableKernelNames(parsedProgram));
            }
            const std::size_t onlyEntryIndex = parsedProgram.entryPoints[0];
            kernelName = parsedProgram.functions[onlyEntryIndex].name;
        }

        std::size_t selectedEntryIndex = 0;
        const PTXFunction* selectedKernel = findKernel(parsedProgram, kernelName, selectedEntryIndex);
        if (selectedKernel == nullptr) {
            throw std::runtime_error(
                "Kernel '" + kernelName + "' was not found. Available kernels: " + availableKernelNames(parsedProgram));
        }

        std::map<std::string, ScalarArgument> scalarLookup;
        for (const ScalarArgument& scalar : scalarArgs) {
            if (!scalarLookup.emplace(scalar.name, scalar).second) {
                throw std::runtime_error("Duplicate scalar parameter provided: " + scalar.name);
            }
        }

        std::map<std::string, PointerArgument> pointerLookup;
        for (const PointerArgument& pointer : pointerArgs) {
            if (!pointerLookup.emplace(pointer.name, pointer).second) {
                throw std::runtime_error("Duplicate pointer parameter provided: " + pointer.name);
            }
        }

        for (const auto& entry : scalarLookup) {
            const bool exists = std::any_of(
                selectedKernel->parameters.begin(),
                selectedKernel->parameters.end(),
                [&](const PTXParameter& parameter) { return parameter.name == entry.first; });
            if (!exists) {
                throw std::runtime_error("Unknown scalar parameter for kernel '" + kernelName + "': " + entry.first);
            }
        }

        for (const auto& entry : pointerLookup) {
            const bool exists = std::any_of(
                selectedKernel->parameters.begin(),
                selectedKernel->parameters.end(),
                [&](const PTXParameter& parameter) { return parameter.name == entry.first; });
            if (!exists) {
                throw std::runtime_error("Unknown pointer parameter for kernel '" + kernelName + "': " + entry.first);
            }
        }

        PTXVM vm;
        if (!vm.initialize()) {
            throw std::runtime_error("Failed to initialize PTX VM");
        }
        if (!vm.loadProgram(filePath.string())) {
            throw std::runtime_error("Failed to load PTX program into the VM");
        }

        PTXProgram executionProgram = parsedProgram;
        executionProgram.entryPoints.clear();
        executionProgram.entryPoints.push_back(selectedEntryIndex);
        if (!vm.getExecutor().initialize(executionProgram)) {
            throw std::runtime_error("Failed to configure the executor for the requested kernel");
        }
        vm.getExecutor().setGridDimensions(
            gridDims[0], gridDims[1], gridDims[2],
            blockDims[0], blockDims[1], blockDims[2]);

        std::vector<KernelParameter> kernelParameters;
        std::vector<PreparedScalar> preparedScalars;
        std::vector<PreparedPointerBuffer> preparedPointers;

        kernelParameters.reserve(selectedKernel->parameters.size());
        preparedScalars.reserve(selectedKernel->parameters.size());
        preparedPointers.reserve(selectedKernel->parameters.size());

        for (const PTXParameter& parameter : selectedKernel->parameters) {
            KernelParameter kernelParameter;
            kernelParameter.size = parameter.size;
            kernelParameter.offset = parameter.offset;

            if (parameter.isPointer) {
                const auto pointerIt = pointerLookup.find(parameter.name);
                if (pointerIt == pointerLookup.end()) {
                    throw std::runtime_error("Missing pointer parameter: " + parameter.name);
                }

                PreparedPointerBuffer buffer;
                buffer.name = parameter.name;
                buffer.bufferType = pointerIt->second.bufferType;
                buffer.elementCount = pointerIt->second.elementCount;
                buffer.beforeBytes = preparePointerBytes(pointerIt->second);
                buffer.byteSize = buffer.beforeBytes.size();
                buffer.deviceAddress = vm.allocateMemory(buffer.byteSize);
                if (buffer.deviceAddress == 0) {
                    throw std::runtime_error("Failed to allocate device memory for parameter: " + parameter.name);
                }
                if (!vm.copyMemoryHtoD(buffer.deviceAddress, buffer.beforeBytes.data(), buffer.beforeBytes.size())) {
                    throw std::runtime_error("Failed to copy host data into device memory for parameter: " + parameter.name);
                }

                kernelParameter.devicePtr = buffer.deviceAddress;
                preparedPointers.push_back(std::move(buffer));
            } else {
                const auto scalarIt = scalarLookup.find(parameter.name);
                if (scalarIt == scalarLookup.end()) {
                    throw std::runtime_error("Missing scalar parameter: " + parameter.name);
                }

                uint64_t packedBits = 0;
                std::string displayValue;
                packScalarBits(parameter, scalarIt->second.value, packedBits, displayValue);
                kernelParameter.devicePtr = packedBits;
                preparedScalars.push_back({parameter.name, parameter.type, displayValue});
            }

            kernelParameters.push_back(kernelParameter);
        }

        vm.setKernelParameters(kernelParameters);
        if (!vm.run()) {
            throw std::runtime_error("Kernel execution failed");
        }

        for (PreparedPointerBuffer& buffer : preparedPointers) {
            buffer.afterBytes.resize(buffer.byteSize, 0);
            if (!vm.copyMemoryDtoH(buffer.afterBytes.data(), buffer.deviceAddress, buffer.afterBytes.size())) {
                throw std::runtime_error("Failed to copy device memory back for parameter: " + buffer.name);
            }
        }

        const std::vector<MemoryWatch> memoryWatches = collectMemoryWatches(vm);
        const std::string logs = capture.combined();

        std::ostringstream out;
        out << "{";
        out << "\"ok\":true,";
        out << "\"kernel\":" << quoteJson(kernelName) << ",";
        out << "\"grid\":[" << gridDims[0] << "," << gridDims[1] << "," << gridDims[2] << "],";
        out << "\"block\":[" << blockDims[0] << "," << blockDims[1] << "," << blockDims[2] << "],";

        out << "\"scalars\":[";
        for (std::size_t i = 0; i < preparedScalars.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const PreparedScalar& scalar = preparedScalars[i];
            out << "{";
            out << "\"name\":" << quoteJson(scalar.name) << ",";
            out << "\"type\":" << quoteJson(scalar.type) << ",";
            out << "\"value\":" << quoteJson(scalar.value);
            out << "}";
        }
        out << "],";

        out << "\"pointer_buffers\":[";
        for (std::size_t i = 0; i < preparedPointers.size(); ++i) {
            if (i != 0) {
                out << ",";
            }

            const PreparedPointerBuffer& buffer = preparedPointers[i];
            const std::size_t previewCount = std::min<std::size_t>(buffer.elementCount, 32);
            const auto beforePreview = decodePreviewValues(buffer.beforeBytes, buffer.bufferType, previewCount);
            const auto afterPreview = decodePreviewValues(buffer.afterBytes, buffer.bufferType, previewCount);

            out << "{";
            out << "\"name\":" << quoteJson(buffer.name) << ",";
            out << "\"buffer_type\":" << quoteJson(buffer.bufferType) << ",";
            out << "\"element_count\":" << buffer.elementCount << ",";
            out << "\"byte_size\":" << buffer.byteSize << ",";
            out << "\"device_address\":" << quoteJson(formatHexAddress(buffer.deviceAddress)) << ",";
            out << "\"preview_count\":" << previewCount << ",";
            out << "\"truncated\":" << (buffer.elementCount > previewCount ? "true" : "false") << ",";
            out << "\"before\":" << joinJsonValueArray(beforePreview) << ",";
            out << "\"after\":" << joinJsonValueArray(afterPreview) << ",";
            out << "\"hex_before\":" << quoteJson(formatHexPreview(buffer.beforeBytes, 32)) << ",";
            out << "\"hex_after\":" << quoteJson(formatHexPreview(buffer.afterBytes, 32));
            out << "}";
        }
        out << "],";

        out << "\"memory_watch\":[";
        for (std::size_t i = 0; i < memoryWatches.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const MemoryWatch& watch = memoryWatches[i];
            const auto preview = decodeInt32Preview(watch.bytes, 16);
            out << "{";
            out << "\"address\":" << quoteJson(formatHexAddress(watch.address)) << ",";
            out << "\"byte_size\":" << watch.bytes.size() << ",";
            out << "\"hex\":" << quoteJson(formatHexPreview(watch.bytes, 32)) << ",";
            out << "\"int32_preview\":" << joinJsonValueArray(preview);
            out << "}";
        }
        out << "],";

        out << "\"logs\":" << quoteJson(logs);
        out << "}";

        std::cout.rdbuf(capture.originalStdout);
        std::cerr.rdbuf(capture.originalStderr);
        capture.originalStdout = nullptr;
        capture.originalStderr = nullptr;
        std::cout << out.str();
        return 0;
    } catch (const std::exception& error) {
        const std::string logs = capture.combined();
        std::cout.rdbuf(capture.originalStdout);
        std::cerr.rdbuf(capture.originalStderr);
        capture.originalStdout = nullptr;
        capture.originalStderr = nullptr;
        std::cout << "{"
                  << "\"ok\":false,"
                  << "\"error\":" << quoteJson(error.what()) << ","
                  << "\"logs\":" << quoteJson(logs)
                  << "}";
        return 1;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            printUsage();
            return 1;
        }

        const std::string command = argv[1];
        const std::filesystem::path filePath = argv[2];
        if (!std::filesystem::exists(filePath)) {
            std::cout << "{"
                      << "\"ok\":false,"
                      << "\"error\":" << quoteJson("PTX file not found: " + filePath.string())
                      << "}";
            return 1;
        }

        if (command == "inspect") {
            return emitInspectJson(filePath);
        }

        if (command != "run") {
            printUsage();
            return 1;
        }

        std::string kernelName;
        std::array<unsigned int, 3> gridDims = {1U, 1U, 1U};
        std::array<unsigned int, 3> blockDims = {32U, 1U, 1U};
        std::vector<ScalarArgument> scalarArgs;
        std::vector<PointerArgument> pointerArgs;

        for (int index = 3; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--kernel") {
                if (index + 1 >= argc) {
                    throw std::runtime_error("--kernel requires a value");
                }
                kernelName = argv[++index];
            } else if (argument == "--grid") {
                if (index + 1 >= argc) {
                    throw std::runtime_error("--grid requires a value");
                }
                gridDims = parseTriplet(argv[++index], "--grid");
            } else if (argument == "--block") {
                if (index + 1 >= argc) {
                    throw std::runtime_error("--block requires a value");
                }
                blockDims = parseTriplet(argv[++index], "--block");
            } else if (argument == "--scalar") {
                if (index + 1 >= argc) {
                    throw std::runtime_error("--scalar requires a value");
                }
                scalarArgs.push_back(parseScalarArgument(argv[++index]));
            } else if (argument == "--pointer") {
                if (index + 1 >= argc) {
                    throw std::runtime_error("--pointer requires a value");
                }
                pointerArgs.push_back(parsePointerArgument(argv[++index]));
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }

        return emitRunJson(filePath, kernelName, gridDims, blockDims, scalarArgs, pointerArgs);
    } catch (const std::exception& error) {
        std::cout << "{"
                  << "\"ok\":false,"
                  << "\"error\":" << quoteJson(error.what())
                  << "}";
        return 1;
    }
}
