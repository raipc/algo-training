plugins {
    kotlin("jvm") version "1.3.72"
    java
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(8))
    }
}